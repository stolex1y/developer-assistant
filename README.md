# Ассистент разработчика

Локальный read-only CLI отвечает на вопросы о Git-проекте по разрешённым tracked
text files: README, `docs/`, схемам/API, исходникам, тестам и другим типам файлов,
допущенным фильтрами. Перед каждым `/help` обновляется in-memory индекс; branch и
список путей поступают от дочернего MCP server по stdio.

MCP server реализован в `src/mcp.rs`: `get_current_branch` и `list_files`.
RAG corpus загружает `src/review/corpus.rs`, а index и provider tools
`search_code`/`read_file` находятся в `src/retrieval.rs` и `src/assistant.rs`.

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
создаётся. `/help` обновляет локальный индекс и MCP-контекст, но не вызывает
provider, пока нет выбранной настроенной модели.

## Запуск и команды

Сначала перейдите в корень Git-проекта, который нужно объяснить:

```bash
cd /path/to/project
cargo run --manifest-path /path/to/developer-assistant/Cargo.toml
```

Текущая директория становится корнем проекта; `--project-root <путь>` остаётся
необязательным override. REPL:

```text
/models
/model deepseek/deepseek-v4-pro
/model default deepseek/deepseek-v4-pro
/help Какие модули есть и где находится web entry point?
/exit
```

Стрелки вверх/вниз перемещаются по введённым `/help`-вопросам только в текущем
процессе. История не записывается на диск. Tab дополняет только `/help`,
`/models`, `/model`, `/exit`; при неоднозначном префиксе показывает варианты.
Аргументы и обычный текст вопроса не дополняются.

`/model <provider>/<model>` меняет модель только для текущей сессии.
`/model default <provider>/<model>` сохраняет default в user config и выбирает
эту модель в текущей сессии. `--model <provider>/<model>` задаёт выбор только
для запущенного процесса и не меняет сохранённый default.

Одноразовый режим `/help` остаётся доступен из корня проекта:

```bash
cd /path/to/project
cargo run --manifest-path /path/to/developer-assistant/Cargo.toml -- \
  --model deepseek/deepseek-flash \
  /help "Как устроена структура проекта?"
```

После `cargo build --release --manifest-path /path/to/developer-assistant/Cargo.toml`
запускайте `target/release/developer-assistant` из корня проекта. `--config` и
`--model` указываются до `/help`.

## Границы и приватность `/help`

- Перед каждым `/help` Git-список tracked-файлов загружается заново. В REPL
  остаётся только in-memory индекс; изменённые/новые документы индексируются,
  удалённые исчезают, неизменённые passages переиспользуются.
- RAG использует до пяти начальных passages и 8 KiB текста. Если корпус
  непустой, но начального совпадения нет, provider всё равно получает read-only
  tools и может запросить контекст. Если допустимых файлов нет, provider не вызывается.
- Допустимый corpus повторяет фильтры PR reviewer: tracked regular files
  (Git modes `100644`/`100755`), без файлов под `.gitignore`, symlinks,
  generated/build/vendor-каталогов, minified/map/lock-файлов, secret-like paths,
  binary/NUL, non-UTF-8 и empty files. Лимит: 10 000 файлов; per-file и aggregate
  source-byte caps отсутствуют. D31 держит corpus/index в памяти, поэтому RAM и
  время растут с объёмом. D32 отдельно сохраняет 256 MiB предел serialized
  embedding cache. Во время успешного `/help` причины пропусков и path examples
  не печатаются; ошибки остаются видимыми. Фильтры и redaction — эвристики, не DLP.
- Локальный BM25 работает по исходникам, тестам, README, docs, схемам и другим
  допустимым tracked text files; Markdown headings влияют на ранжирование, но
  остаются метаданными, а не частью body excerpt. Vector DB и embeddings для `/help`
  не используются.
- Внутренний MCP-контекст содержит текущую ветку и до 50 уникальных tracked paths;
  этот список не выводится перед успешным ответом.
  Model tools — только `search_code` и `read_file`; они обращаются к тому же
  отфильтрованному in-memory corpus. `search_code` возвращает до восьми passages
  и 8 KiB; `read_file` ограничен 80 строками и 8 KiB. Shell, write и diff tools
  отсутствуют.
- Ответ сохраняет inline citations модели, но приложение не добавляет отдельный
  footer `Источники:` и не собирает список source references для печати.
- Ответ `/help` рендерится как Markdown: headings, emphasis, ordered/unordered
  lists, inline/fenced code, links и tables. Таблицы шире терминала и все таблицы
  со ссылками выводятся как строки «заголовок: значение» с переносом по ширине,
  чтобы URL не ломали выравнивание. Остальной текст переносится по доступной ширине;
  строки кода сохраняются без переноса и могут быть шире терминала. ANSI-оформление
  включается только для TTY. URL ссылок всегда видны рядом с подписью; OSC8
  hyperlinks не отправляются в терминал. Redirected output не содержит ANSI.
- MCP subprocess не наследует переменные окружения с ключами настроенных providers.
- Provider получает вопрос, ограниченный MCP-контекст, начальные passages и
  только запрошенные bounded tool results. Полный README, индекс и исходники
  целиком не отправляются. Успешный вывод не печатает MCP inventory или счётчики
  индекса; остаются ответ и token-budget line.
- До retrieval удаляются значения с поддерживаемыми credential-метками после `:` или
  `=`: API key/token, access/refresh token, secret, password, OTP, phone,
  authorization, bearer и несколько русских эквивалентов. Цитируемые значения
  могут содержать пробелы; нецитируемое значение заканчивается на `;` или конце
  строки. Не помещайте реальные секреты и финансовые данные в проект.
- API key не печатается и не сохраняется. Provider errors не включают raw body.
  Запросы и найденные excerpts покидают компьютер при remote provider call.
- Tool loop суммирует uncached input tokens ответов с включёнными tools до
  30 000 на один `/help`; фиксированного call-count limit нет. После достижения
  лимита новые tools не исполняются; один финальный запрос отправляется без
  tools. Его usage проверяется, показывается отдельно и не прибавляется к loop
  budget. Отсутствующие или противоречивые counters останавливают запрос без
  оценки токенов и fallback.
- При этом token-only limit не ограничивает provider, который многократно возвращает новые tool calls с нулевыми uncached input tokens. Отдельного ограничения числа вызовов нет по согласованному контракту.
- Перед выводом экранируются terminal controls и bidi-direction marks в ответе,
  Git branch, путях и source citations; переводы строк в ответе сохраняются.

Provider contract: OpenAI-compatible `POST /chat/completions`, function calling,
`temperature=0`, `max_tokens=4096`, timeout 60 секунд. Если финальный completion
имеет `finish_reason=length`, `/help` явно помечает ответ как неполный и не
повторяет запрос. Каждый ответ должен содержать `prompt_tokens` и
`prompt_tokens_details` с согласованными cache-hit / cache-miss counters
(`cached_tokens` — допустимый alias). DeepSeek profile в `config.example.toml`
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
Для учёта нужны `prompt_tokens` и хотя бы один cache counter. Если provider
возвращает `prompt_cache_miss_tokens`, используется он; иначе uncached count
вычисляется как `prompt_tokens - cache-hit count` (`prompt_cache_hit_tokens` или
`cached_tokens`). Отсутствие обоих counters или противоречивые значения
останавливают review без публикации.

В целевом GitHub repository задайте Actions secret `DEEPSEEK_API_KEY`.
`GITHUB_TOKEN` workflow получает автоматически с `contents: write` и
`pull-requests: write`: первое разрешение даёт job техническую возможность менять
содержимое репозитория, второе требуется для PR comments и разрешения review threads.
Утилита сама не выполняет commit, push или merge.
Workflow запускается при `opened`, `reopened` и `synchronize`; результат — PR summary
и обновляемые inline threads.

В PR summary и inline comments попадают findings с path и evidence, подтверждёнными
diff. Для добавленных/изменённых файлов evidence берётся из added/current-side; для
удалённых допустимо removed-side evidence только при регрессии от удаления, и такой
finding выводится только в summary без inline anchor. Неподтверждённые findings
исключаются; summary показывает их число. Их исключение не блокирует отдельные
resolutions: прежний thread закрывается только при совпадении fingerprint/path и
evidence в добавленном коде.

Краткое AI-резюме не проверяется отдельно. У findings сверяются path/evidence с diff;
смысл замечаний требует ручной оценки.

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
