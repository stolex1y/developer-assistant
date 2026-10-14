# Ассистент разработчика

Локальный read-only CLI отвечает на вопросы о структуре проекта по корневому
`README.md` и Markdown-документам в `docs/`. Текущая Git-ветка и список путей
поступают через локальный MCP server.

## Требования

- Rust 1.89 или новее;
- Git в `PATH`;
- API key и доступ к OpenAI-compatible Chat Completions provider для ответов модели.

## Настройка providers и моделей

Пример `config.example.toml` задаёт DeepSeek и две модели. Скопируйте его в
пользовательский каталог и введите ключ скрыто, без записи значения в историю
команд:

```bash
mkdir -p ~/.config/developer-assistant
cp -n config.example.toml ~/.config/developer-assistant/config.toml
read -rs -p "DeepSeek API key: " DEEPSEEK_API_KEY
printf '\n'
export DEEPSEEK_API_KEY
```

По умолчанию CLI читает `~/.config/developer-assistant/config.toml`. Другой
путь можно передать через `--config <путь>`. Config хранит base URL, список
разрешённых model IDs и имя переменной окружения `api_key_env`, но не значение
ключа. Для каждого provider можно задать отдельную переменную. Модели берутся
только из конфигурации; сетевого автообнаружения нет.

DeepSeek endpoint и model IDs сверены с [официальной спецификацией Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion) и [историей обновлений моделей](https://api-docs.deepseek.com/updates).

Для нового provider добавьте его в TOML:

```toml
[providers.example]
base_url = "https://api.example.com/v1"
api_key_env = "EXAMPLE_API_KEY"
models = ["model-a", "model-b"]

[selection]
default_model = "example/model-a"
```

Удалённый endpoint должен использовать HTTPS. HTTP разрешён только для loopback
fake provider в тестах. Redirects не выполняются.

Если config отсутствует, приложение всё равно запускает REPL. `/models` покажет
пустой список, путь config и `config.example.toml`; файл автоматически не
создаётся. Пока нет выбранной настроенной модели, `/help` не вызывает provider.

## Запуск и команды

Откройте REPL для текущего Git-проекта:

```bash
cargo run -- --project-root /path/to/project
```

Если `--project-root` пропущен, используется текущая директория. Она должна быть
корнем Git working tree. Команды REPL:

```text
/models
/model deepseek/deepseek-v4-pro
/model default deepseek/deepseek-v4-pro
/help Какие модули есть и где находится web entry point?
/exit
```

`/model <provider>/<model>` меняет модель только для текущей сессии.
`/model default <provider>/<model>` сохраняет default в user config и выбирает
эту модель в текущей сессии. `--model <provider>/<model>` задаёт выбор только
для запущенного процесса и не меняет сохранённый default.

Одноразовый режим `/help` остаётся доступен:

```bash
cargo run -- --project-root /path/to/project \
  --model deepseek/deepseek-flash \
  /help "Как устроена структура проекта?"
```

После `cargo build --release` аналогичные команды запускаются через
`target/release/developer-assistant`. `--config` и `--model` указываются до
`/help`.

## Границы и приватность `/help`

- RAG читает только корневой `README.md` и Markdown-файлы из `docs/`.
  Исходный код, `config/`, `.env`, runtime state, examples и fixtures не сканируются.
- Поиск локальный, секционный BM25; используется не более пяти passages и 8 KiB
  текста на запрос. Vector DB и embeddings не используются.
- MCP-инструменты доступны только для чтения: текущая ветка и не более 50
  уникальных путей из Git index (`--deduplicate`). Неотслеживаемые пути (включая
  подпадающие под `.gitignore`), содержимое файлов, diff и команды модели исключены.
- MCP subprocess не наследует переменные окружения с ключами настроенных providers.
- Provider получает пользовательский вопрос, найденные excerpts и branch/path
  context. Полный README, полный file tree и исходники не отправляются.
- До retrieval удаляются значения с поддерживаемыми credential-метками после `:` или
  `=`: API key/token, access/refresh token, secret, password, OTP, phone,
  authorization, bearer и несколько русских эквивалентов. Цитируемые значения
  могут содержать пробелы; нецитируемое значение заканчивается на `;` или конце
  строки. Это эвристика, а не DLP: не помещайте реальные секреты и финансовые
  данные в README или `docs/`.
- API key не печатается и не сохраняется. Provider errors не включают raw body.
  Учитывайте, что запрос и отобранные excerpts покидают компьютер при remote
  provider call.
- Model получает только system/user messages, без tools; результат не выполняется
  и не меняет проект.
- Перед выводом экранируются terminal controls и bidi-direction marks в ответе,
  Git branch, путях и source headings; обычные переводы строки в ответе сохраняются.

Provider contract: OpenAI-compatible `POST /chat/completions`, `temperature=0`,
`max_tokens=800`, timeout 60 секунд. DeepSeek profile в `config.example.toml`
использует официальный base URL и model IDs `deepseek-flash` и `deepseek-v4-pro`.

## Автоматическое ревью pull request

Reusable workflow хранится в `.github/workflows/review.yml`; проекты подключают
его через caller workflow, закреплённый на неизменяемом commit SHA. Сборка и
запуск идут на `ubuntu-24.04`. Утилита компилируется из доверенного utility
checkout вне PR workspace; исходный код PR не запускается и не изменяется.
Fork pull requests пропускаются.

Review использует отдельный filtered RAG по tracked text-документации и коду.
Embeddings строятся локально через `fastembed-rs` 6.1.0 и ONNX Runtime CPU на
неквантованной `AllMiniLML6V2` (384 измерения). Batch size — 8 passages; модель
сохраняет весь допустимый corpus, ограничивая пиковую память inference.
Ollama, PyTorch, daemon и GPU не нужны. Actions cache хранит Cargo registry,
модель и content-validated индекс в игнорируемом `.data/developer-assistant-review`.
Скомпилированный `target/` не кэшируется. При первом запуске runner скачивает
зависимости и веса; для этого требуется доступ к crates.io и Hugging Face.

DeepSeek получает только допустимый PR diff, отобранные passages и краткие
предыдущие findings. Полный индекс, credentials и посторонние файлы не
отправляются. По умолчанию caller задаёт лимит 30 000 uncached input tokens для
tool-enabled запросов; usage финального запроса без tools показывается отдельно.
Изменить лимит можно в `with.max_uncached_input_tokens` caller workflow.
Для проверки согласованности нужны `prompt_tokens` и `prompt_cache_miss_tokens`;
cache-hit count читается из `prompt_cache_hit_tokens` или `cached_tokens`, а если
его нет — вычисляется как разница. Отсутствующий uncached usage или противоречивые
значения останавливают review без публикации.

В целевом GitHub repository задайте Actions secret `DEEPSEEK_API_KEY`.
`GITHUB_TOKEN` workflow получает автоматически с `contents: read` и
`pull-requests: write`. Утилита запускается при `opened`, `reopened` и
`synchronize`; результат — PR summary и обновляемые inline threads.

**Граница доверия:** включайте этот workflow только для репозитория, где
пользователям с правом менять workflows можно доверять repository secrets.
`pull_request` может использовать caller workflow из PR merge commit; автор
same-repository PR, которому разрешено менять `.github/workflows`, способен
изменить и сам caller. Утилита не запускает PR source code, но эта схема не
защищает секрет от злонамеренного редактора workflow. Fork PR не получают ключ
и пропускаются; не предоставляйте write access недоверенным участникам.

Перед реальным запуском опубликуйте utility commit и caller commit; caller
должен ссылаться на опубликованный immutable SHA. Проверка без credentials и
внешних API:

```bash
cargo test --locked --test review_acceptance
```

## Проверки

```bash
cargo fmt --all -- --check
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

Тесты используют локальный fake HTTP provider и не требуют реального ключа или
сетевого вызова к модели.
