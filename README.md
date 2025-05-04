
 📦 Репозиторий: glaza-boga-bot

### 👁️ Glaza Boga — Интеллектуальный Telegram-бот для поиска информации

---

### Описание

**Glaza Boga** — это многофункциональный Telegram-бот для поиска информации по различным базам данных, с поддержкой семантического поиска на базе ML, системой тикетов поддержки, VIP-статусами, админ-панелью и гибкой системой лимитов.

Бот предназначен для быстрого и удобного поиска по CSV, Excel, JSON и текстовым файлам, а также для автоматизации поддержки пользователей через тикеты.

---

### 🚀 Основные возможности

- **Универсальный поиск** по различным форматам данных (CSV, Excel, JSON, TXT)
- **Семантический поиск** с использованием Sentence Transformers
- **Личный кабинет пользователя** (статистика, история запросов, VIP-статус)
- **Система тикетов поддержки** (создание, ответы, админ-панель)
- **Гибкая система лимитов** (free, VIP, admin)
- **Админ-панель** (статистика, рассылки, управление VIP и банами)
- **Автоматическое обновление и бэкапы баз данных**
- **Логирование событий и ошибок**

---

### 🛠️ Технологии

- Python 3.8+
- [pyTelegramBotAPI (telebot)](https://github.com/eternnoir/pyTelegramBotAPI)
- [pandas](https://pandas.pydata.org/)
- [sentence-transformers](https://www.sbert.net/)
- [scikit-learn](https://scikit-learn.org/)
- [openpyxl](https://openpyxl.readthedocs.io/)
- [sqlite3](https://docs.python.org/3/library/sqlite3.html)
- [logging](https://docs.python.org/3/library/logging.html)

---

### 📦 Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/yourusername/glaza-boga-bot.git
   cd glaza-boga-bot
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Укажите свой Telegram Bot Token в файле конфигурации или переменной окружения.

4. Запустите бота:
   ```bash
   python testGlaza.py
   ```

---

### 📝 Пример использования

- Запустите бота в Telegram.
- Используйте команды `/start` или `/menu` для вызова главного меню.
- Выбирайте тип поиска, отправляйте запросы, скачивайте отчеты.
- Открывайте тикеты поддержки и управляйте ими через меню.

---

### 🧑‍💻 Вклад

Pull requests приветствуются! Открывайте issues для багов и предложений.

---



---

### 🤝 Благодарности

- [QVENTIS_TEAM](https://t.me/QVENTIS_TEAM) — поддержка и идеи
- [SBERT](https://www.sbert.net/) — за отличные ML-модели

---



