# Ассистент разработчика

Локальный read-only CLI отвечает на вопросы о структуре проекта по его
`README.md` и Markdown-документам в `docs/`. Актуальные Git branch и список
путей поступают через локальный MCP server.

## Требования

- Rust 1.89 или новее;
- Git в `PATH`;
- доступ к выбранному OpenAI-compatible Chat Completions provider.

## Настройка provider

Ключ обязателен и читается только из окружения. По умолчанию используются
OpenAI API и модель `gpt-4o-mini`:

```bash
export DEVELOPER_ASSISTANT_API_KEY='ваш-ключ'
# Необязательно:
export DEVELOPER_ASSISTANT_BASE_URL='https://api.openai.com/v1'
export DEVELOPER_ASSISTANT_MODEL='gpt-4o-mini'
```

`DEVELOPER_ASSISTANT_BASE_URL` — base URL, к которому CLI добавляет
`/chat/completions`. Remote HTTP endpoints без TLS запрещены; HTTP разрешён только
для loopback fake provider. Redirects не выполняются. CLI не читает `.env` и не
сохраняет ключ.

## Запуск

Из этого репозитория:

```bash
cargo run -- --project-root /path/to/project /help "Какие модули есть и где находится web entry point?"
```

После `cargo build --release`:

```bash
target/release/developer-assistant --project-root /path/to/project /help "Как устроена структура проекта?"
```

Если `--project-root` пропущен, используется текущая директория. Она должна быть
корнем Git working tree. Команда `/help` принимает вопрос целиком; CLI выводит
ответ, текущую ветку, ограниченный список файлов и источники с заголовками.

## Границы и приватность

- RAG читает только корневой `README.md` и Markdown-файлы из `docs/`.
  Исходный код, `config/`, `.env`, runtime state, examples и fixtures не сканируются.
- Поиск локальный, секционный BM25; используется не более пяти passages и 8 KiB
  текста на запрос. Vector DB и embeddings не используются.
- MCP-инструменты доступны только для чтения: текущая ветка и не более 50
  уникальных путей из Git index (`--deduplicate`). Неотслеживаемые пути (включая
  подпадающие под `.gitignore`), содержимое файлов, diff и команды модели исключены.
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

Фиксированный API contract: OpenAI-compatible `POST /chat/completions`, модель по
умолчанию `gpt-4o-mini`, `temperature=0`, `max_completion_tokens=800`. Другую
совместимую base URL/model можно задать переменными окружения.

## Проверки

```bash
cargo fmt --all -- --check
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

Тесты используют локальный fake HTTP provider и не требуют реального ключа или
сетевого вызова к модели.