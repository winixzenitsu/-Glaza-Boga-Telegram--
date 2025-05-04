import os
import sqlite3
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import telebot
from telebot import types
from telebot.apihelper import ApiTelegramException
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chardet
import hashlib
import traceback
import json
import openpyxl
import logging
import re
import pytz
from io import BytesIO
import zipfile
import sys
import io

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    def __init__(self):
        # Основные настройки
        self.token = "ваш токен бота"
        self.data_folder = "database"
        self.backup_folder = "backups"
        self.reports_folder = "reports"
        self.support_folder = "support_tickets"
        self.admin_id = [ваш тг айди]
        self.vip_users = []
        self.bot_username = "юзернейм вашего бота"
        
        # Лимиты
        self.daily_limit = {
            "free": 3,
            "vip": 15,
            "admin": 1000
        }
        
        # Настройки модели
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.max_results = 50
        self.similarity_threshold = 0.3
        
        # Другие настройки
        self.auto_update = True
        self.search_timeout = 10
        self.backup_interval = 86400  # 24 часа
        self.vip_price = 500
        self.timezone = pytz.timezone('Europe/Moscow')
        self.support_chat_id = -1001234567890  # ID чата для поддержки
        self.max_tickets_per_user = 3  # Максимальное количество открытых тикетов
        self.ticket_expiry_days = 7  # Дней до автоматического закрытия тикета

CONFIG = Config()

# ==================== ИНИЦИАЛИЗАЦИЯ БОТА ====================
try:
    bot = telebot.TeleBot(CONFIG.token)
    CONFIG.bot_username = bot.get_me().username
except Exception as e:
    logger.error(f"Ошибка инициализации бота: {e}")
    raise

# ==================== ФИЛЬТР ДЛЯ АКТУАЛЬНЫХ СООБЩЕНИЙ ====================
def is_actual_message(message):
    # Только личные чаты (private) и только новые сообщения
    if message.chat.type != 'private':
        return False
    # Можно добавить фильтр по времени, если нужно
    return True

# ==================== МОДЕЛЬ ML ====================
model_lock = threading.Lock()
model = None
model_loading = False
model_loaded = False
model_load_start = None

def load_model():
    """Загрузка модели ML"""
    global model, model_loading, model_loaded, model_load_start
    with model_lock:
        model_loading = True
        model_load_start = time.time()
        try:
            logger.info("Начало загрузки модели...")
            model = SentenceTransformer(CONFIG.model_name)
            model_loaded = True
            load_time = time.time() - model_load_start
            logger.info(f"✅ Модель загружена за {load_time:.2f} сек")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            model = None
            model_loaded = False
        finally:
            model_loading = False

# ==================== БАЗА ДАННЫХ ====================
class DatabaseManager:
    """Класс для управления базой данных бота"""
    
    def __init__(self):
        self.db_path = os.path.join(CONFIG.data_folder, "user_data.db")
        self.lock = threading.Lock()
        self.conn = None
        self.cursor = None
        self._initialize_db()
        self._run_periodic_tasks()

    def _initialize_db(self):
        """Инициализация структуры базы данных"""
        try:
            os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            
            # Таблица пользователей
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    requests_today INTEGER DEFAULT 0,
                    last_request_date TEXT,
                    join_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_vip BOOLEAN DEFAULT FALSE,
                    balance INTEGER DEFAULT 0,
                    vip_expiry_date TEXT,
                    banned BOOLEAN DEFAULT FALSE
                );
            """)
            
            # Таблица истории поиска
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    query TEXT,
                    date TEXT DEFAULT CURRENT_TIMESTAMP,
                    results_count INTEGER,
                    search_type TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
            """)
            
            # Таблица платежей
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    amount INTEGER,
                    date TEXT DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,
                    payment_method TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
            """)
            
            # Таблица тикетов поддержки
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS support_tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    ticket_id TEXT,
                    subject TEXT,
                    message TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    admin_id INTEGER,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
            """)
            
            # Таблица сообщений поддержки
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS support_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT,
                    user_id INTEGER,
                    message TEXT,
                    is_admin BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(ticket_id) REFERENCES support_tickets(ticket_id)
                );
            """)
            
            self.conn.commit()
            logger.info(f"✅ База данных инициализирована: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            if self.conn:
                self.conn.rollback()
            raise

    def _run_periodic_tasks(self):
        """Запуск периодических задач"""
        def task():
            while True:
                time.sleep(3600)  # Каждый час
                self._check_vip_expiry()
                self._cleanup_old_data()
                self._close_expired_tickets()
                
        threading.Thread(target=task, daemon=True).start()

    def _check_vip_expiry(self):
        """Проверка истечения VIP статуса"""
        with self.lock:
            try:
                today = datetime.now(CONFIG.timezone).strftime('%Y-%m-%d')
                self.cursor.execute(
                    "UPDATE users SET is_vip = FALSE WHERE vip_expiry_date < ? AND is_vip = TRUE",
                    (today,)
                )
                if self.cursor.rowcount > 0:
                    logger.info(f"Снят VIP статус у {self.cursor.rowcount} пользователей")
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Ошибка проверки VIP статусов: {e}")

    def _cleanup_old_data(self):
        """Очистка старых данных"""
        with self.lock:
            try:
                month_ago = (datetime.now(CONFIG.timezone) - timedelta(days=30)).strftime('%Y-%m-%d')
                self.cursor.execute(
                    "DELETE FROM search_history WHERE date < ?",
                    (month_ago,)
                )
                if self.cursor.rowcount > 0:
                    logger.info(f"Удалено {self.cursor.rowcount} старых записей истории")
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Ошибка очистки старых данных: {e}")

    def _close_expired_tickets(self):
        """Закрытие просроченных тикетов"""
        with self.lock:
            try:
                expiry_date = (datetime.now(CONFIG.timezone) - timedelta(days=CONFIG.ticket_expiry_days)).strftime('%Y-%m-%d')
                self.cursor.execute(
                    "UPDATE support_tickets SET status = 'closed' WHERE status = 'open' AND created_at < ?",
                    (expiry_date,)
                )
                if self.cursor.rowcount > 0:
                    logger.info(f"Закрыто {self.cursor.rowcount} просроченных тикетов")
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Ошибка закрытия просроченных тикетов: {e}")

    def add_user(self, user: types.User) -> None:
        """Добавление нового пользователя"""
        with self.lock:
            try:
                self.cursor.execute("""
                    INSERT OR IGNORE INTO users (user_id, username, first_name, last_name)
                    VALUES (?, ?, ?, ?)
                """, (user.id, user.username, user.first_name, user.last_name))
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Ошибка добавления пользователя: {e}")
                self.conn.rollback()

    def get_user(self, user_id: int) -> Optional[Dict]:
        """Получение информации о пользователе"""
        with self.lock:
            try:
                self.cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = self.cursor.fetchone()
                if row:
                    columns = [col[0] for col in self.cursor.description]
                    return dict(zip(columns, row))
                return None
            except sqlite3.Error as e:
                logger.error(f"Ошибка получения пользователя: {e}")
                return None

    def update_request_stats(self, user_id: int, query: str, results_count: int, search_type: str = "universal") -> None:
        """Обновление статистики запросов"""
        with self.lock:
            try:
                today = datetime.now(CONFIG.timezone).strftime('%Y-%m-%d')
                user = self.get_user(user_id)
                
                if user and user['last_request_date'] != today:
                    self.cursor.execute("""
                        UPDATE users 
                        SET requests_today = 1, 
                            last_request_date = ?
                        WHERE user_id = ?
                    """, (today, user_id))
                else:
                    self.cursor.execute("""
                        UPDATE users 
                        SET requests_today = requests_today + 1,
                            last_request_date = ?
                        WHERE user_id = ?
                    """, (today, user_id))
                
                self.cursor.execute("""
                    INSERT INTO search_history (user_id, query, results_count, search_type)
                    VALUES (?, ?, ?, ?)
                """, (user_id, query, results_count, search_type))
                
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Ошибка обновления статистики: {e}")
                self.conn.rollback()

    def check_request_limit(self, user_id: int) -> bool:
        """Проверка лимита запросов"""
        user = self.get_user(user_id)
        if not user:
            return False
            
        if user.get('banned', False):
            return False
            
        if user_id in CONFIG.admin_id:
            limit = CONFIG.daily_limit['admin']
        elif user['is_vip'] or user_id in CONFIG.vip_users:
            limit = CONFIG.daily_limit['vip']
        else:
            limit = CONFIG.daily_limit['free']
            
        return user['requests_today'] < limit

    def get_search_history(self, user_id: int, limit: int = 5) -> List[Dict]:
        """Получение истории поиска"""
        with self.lock:
            try:
                self.cursor.execute("""
                    SELECT query, date, results_count, search_type 
                    FROM search_history 
                    WHERE user_id = ? 
                    ORDER BY date DESC 
                    LIMIT ?
                """, (user_id, limit))
                
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"Ошибка получения истории поиска: {e}")
                return []

    def get_full_stats(self) -> Dict:
        """Получение полной статистики"""
        with self.lock:
            try:
                stats = {}
                
                # Общая статистика
                self.cursor.execute("SELECT COUNT(*) FROM users")
                stats['total_users'] = self.cursor.fetchone()[0]
                
                self.cursor.execute("SELECT COUNT(*) FROM users WHERE is_vip = TRUE")
                stats['vip_users'] = self.cursor.fetchone()[0]
                
                self.cursor.execute("SELECT COUNT(*) FROM search_history")
                stats['total_searches'] = self.cursor.fetchone()[0]
                
                self.cursor.execute("SELECT COUNT(*) FROM support_tickets")
                stats['total_tickets'] = self.cursor.fetchone()[0]
                
                # Популярные запросы
                self.cursor.execute("""
                    SELECT query, COUNT(*) as count 
                    FROM search_history 
                    GROUP BY query 
                    ORDER BY count DESC 
                    LIMIT 5
                """)
                stats['top_queries'] = self.cursor.fetchall()
                
                return stats
            except sqlite3.Error as e:
                logger.error(f"Ошибка получения статистики: {e}")
                return {}

    def add_vip(self, user_id: int, days: int = 30) -> bool:
        """Добавление VIP статуса"""
        with self.lock:
            try:
                expiry_date = (datetime.now(CONFIG.timezone) + timedelta(days=days)).strftime('%Y-%m-%d')
                self.cursor.execute("""
                    UPDATE users 
                    SET is_vip = TRUE, 
                        vip_expiry_date = ?
                    WHERE user_id = ?
                """, (expiry_date, user_id))
                self.conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Ошибка добавления VIP: {e}")
                self.conn.rollback()
                return False

    def remove_vip(self, user_id: int) -> bool:
        """Удаление VIP статуса"""
        with self.lock:
            try:
                self.cursor.execute("""
                    UPDATE users 
                    SET is_vip = FALSE,
                        vip_expiry_date = NULL
                    WHERE user_id = ?
                """, (user_id,))
                self.conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Ошибка удаления VIP: {e}")
                self.conn.rollback()
                return False

    def ban_user(self, user_id: int) -> bool:
        """Блокировка пользователя"""
        with self.lock:
            try:
                self.cursor.execute("""
                    UPDATE users 
                    SET banned = TRUE
                    WHERE user_id = ?
                """, (user_id,))
                self.conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Ошибка блокировки пользователя: {e}")
                self.conn.rollback()
                return False

    def unban_user(self, user_id: int) -> bool:
        """Разблокировка пользователя"""
        with self.lock:
            try:
                self.cursor.execute("""
                    UPDATE users 
                    SET banned = FALSE
                    WHERE user_id = ?
                """, (user_id,))
                self.conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Ошибка разблокировки пользователя: {e}")
                self.conn.rollback()
                return False

    def create_support_ticket(self, user_id: int, subject: str, message: str) -> Optional[str]:
        """Создание тикета поддержки"""
        with self.lock:
            try:
                # Проверяем количество открытых тикетов
                self.cursor.execute("""
                    SELECT COUNT(*) FROM support_tickets 
                    WHERE user_id = ? AND status = 'open'
                """, (user_id,))
                open_tickets = self.cursor.fetchone()[0]
                
                if open_tickets >= CONFIG.max_tickets_per_user:
                    return None
                
                ticket_id = hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest()[:8]
                
                self.cursor.execute("""
                    INSERT INTO support_tickets (user_id, ticket_id, subject, message)
                    VALUES (?, ?, ?, ?)
                """, (user_id, ticket_id, subject, message))
                
                self.conn.commit()
                return ticket_id
            except sqlite3.Error as e:
                logger.error(f"Ошибка создания тикета: {e}")
                self.conn.rollback()
                return None

    def add_support_message(self, ticket_id: str, user_id: int, message: str, is_admin: bool = False) -> bool:
        """Добавление сообщения в тикет"""
        with self.lock:
            try:
                self.cursor.execute("""
                    INSERT INTO support_messages (ticket_id, user_id, message, is_admin)
                    VALUES (?, ?, ?, ?)
                """, (ticket_id, user_id, message, is_admin))
                
                # Обновляем дату изменения тикета
                self.cursor.execute("""
                    UPDATE support_tickets 
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE ticket_id = ?
                """, (ticket_id,))
                
                self.conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Ошибка добавления сообщения в тикет: {e}")
                self.conn.rollback()
                return False

    def get_ticket(self, ticket_id: str) -> Optional[Dict]:
        """Получение информации о тикете"""
        with self.lock:
            try:
                self.cursor.execute("""
                    SELECT * FROM support_tickets WHERE ticket_id = ?
                """, (ticket_id,))
                row = self.cursor.fetchone()
                if row:
                    columns = [col[0] for col in self.cursor.description]
                    return dict(zip(columns, row))
                return None
            except sqlite3.Error as e:
                logger.error(f"Ошибка получения тикета: {e}")
                return None

    def get_ticket_messages(self, ticket_id: str) -> List[Dict]:
        """Получение сообщений тикета"""
        with self.lock:
            try:
                self.cursor.execute("""
                    SELECT * FROM support_messages 
                    WHERE ticket_id = ? 
                    ORDER BY created_at
                """, (ticket_id,))
                
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"Ошибка получения сообщений тикета: {e}")
                return []

    def get_user_tickets(self, user_id: int) -> List[Dict]:
        """Получение тикетов пользователя"""
        with self.lock:
            try:
                self.cursor.execute("""
                    SELECT * FROM support_tickets 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC
                """, (user_id,))
                
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"Ошибка получения тикетов пользователя: {e}")
                return []

    def get_open_tickets(self) -> List[Dict]:
        """Получение открытых тикетов"""
        with self.lock:
            try:
                self.cursor.execute("""
                    SELECT * FROM support_tickets 
                    WHERE status = 'open' 
                    ORDER BY created_at
                """)
                
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"Ошибка получения открытых тикетов: {e}")
                return []

    def update_ticket_status(self, ticket_id: str, status: str, admin_id: Optional[int] = None) -> bool:
        """Обновление статуса тикета"""
        with self.lock:
            try:
                if admin_id:
                    self.cursor.execute("""
                        UPDATE support_tickets 
                        SET status = ?, admin_id = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE ticket_id = ?
                    """, (status, admin_id, ticket_id))
                else:
                    self.cursor.execute("""
                        UPDATE support_tickets 
                        SET status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE ticket_id = ?
                    """, (status, ticket_id))
                
                self.conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Ошибка обновления статуса тикета: {e}")
                self.conn.rollback()
                return False

    def close(self):
        """Закрытие соединения с базой данных"""
        if self.conn:
            self.conn.close()
            logger.info("Соединение с базой данных закрыто")

# Инициализация базы данных
user_db = DatabaseManager()

# ==================== ПОИСКОВАЯ СИСТЕМА ====================
class DataSearcher:
    """Класс для поиска по различным форматам данных"""
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.databases = {}
        self.embeddings = {}
        self.lock = threading.Lock()
        self.last_update = None
        
        os.makedirs(self.data_folder, exist_ok=True)
        self.load_all_databases()
        
        if CONFIG.auto_update:
            threading.Thread(target=self.auto_update, daemon=True).start()
            threading.Thread(target=self.periodic_reindex, daemon=True).start()

    def periodic_reindex(self):
        """Периодическая переиндексация данных"""
        while True:
            time.sleep(3600 * 6)  # Каждые 6 часов
            with self.lock:
                if model_loaded:
                    self._reindex_embeddings()

    def _reindex_embeddings(self):
        """Переиндексация эмбеддингов"""
        try:
            logger.info("Начало переиндексации эмбеддингов...")
            for filename, db_info in self.databases.items():
                if db_info['type'] == 'table':
                    text_cols = [col for col in db_info['data'].columns 
                               if pd.api.types.is_string_dtype(db_info['data'][col])]
                    if text_cols:
                        texts = db_info['data'][text_cols].astype(str).apply(' '.join, axis=1).tolist()
                        self.embeddings[filename] = model.encode(texts, show_progress_bar=False)
            logger.info("✅ Переиндексация эмбеддингов завершена")
        except Exception as e:
            logger.error(f"Ошибка переиндексации: {e}")

    def load_all_databases(self) -> None:
        """Загрузка всех баз данных из папки"""
        with self.lock:
            self.databases.clear()
            self.embeddings.clear()
            
            if not os.path.exists(self.data_folder):
                os.makedirs(self.data_folder)
                logger.info(f"Создана папка для баз данных: {self.data_folder}")
                return
                
            for root, _, files in os.walk(self.data_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if file.endswith('.csv'):
                            self._load_csv(file_path)
                        elif file.endswith(('.xlsx', '.xls')):
                            self._load_excel(file_path)
                        elif file.endswith('.json'):
                            self._load_json(file_path)
                        elif file.endswith('.txt'):
                            self._load_txt(file_path)
                        logger.info(f"Загружена база: {file}")
                    except Exception as e:
                        logger.error(f"Ошибка загрузки {file}: {str(e)[:200]}")
            
            self.last_update = datetime.now(CONFIG.timezone)
            logger.info(f"✅ Все базы данных загружены. Всего: {len(self.databases)}")

    def _load_csv(self, file_path: str) -> None:
        """Загрузка CSV файла"""
        try:
            for encoding in ['utf-8', 'windows-1251', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', dtype=str)
                    self._process_dataframe(df, file_path)
                    return
                except UnicodeDecodeError:
                    continue
            logger.error(f"Не удалось определить кодировку файла: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки CSV {file_path}: {e}")

    def _load_excel(self, file_path: str) -> None:
        """Загрузка Excel файла"""
        try:
            df = pd.read_excel(file_path, engine='openpyxl', dtype=str)
            self._process_dataframe(df, file_path)
        except Exception as e:
            logger.error(f"Ошибка загрузки Excel {file_path}: {e}")

    def _load_json(self, file_path: str) -> None:
        """Загрузка JSON файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
            
            self._process_dataframe(df, file_path)
        except Exception as e:
            logger.error(f"Ошибка загрузки JSON {file_path}: {e}")

    def _load_txt(self, file_path: str) -> None:
        """Загрузка текстового файла"""
        try:
            for encoding in ['utf-8', 'windows-1251', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    self.databases[os.path.basename(file_path)] = {
                        'type': 'text',
                        'content': content,
                        'path': file_path,
                        'hash': self._file_hash(file_path)
                    }
                    return
                except UnicodeDecodeError:
                    continue
            logger.error(f"Не удалось определить кодировку файла: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки TXT {file_path}: {e}")

    def _process_dataframe(self, df: pd.DataFrame, file_path: str) -> None:
        """Обработка DataFrame"""
        try:
            df = df.dropna(axis=1, how='all')
            df = df.fillna('')
            
            filename = os.path.basename(file_path)
            self.databases[filename] = {
                'type': 'table',
                'data': df,
                'path': file_path,
                'hash': self._file_hash(file_path)
            }
            
            if model_loaded:
                self._index_dataframe(df, filename)
        except Exception as e:
            logger.error(f"Ошибка обработки DataFrame {file_path}: {e}")

    def _index_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """Индексация DataFrame для семантического поиска"""
        try:
            text_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
            if text_cols:
                texts = df[text_cols].astype(str).apply(' '.join, axis=1).tolist()
                self.embeddings[filename] = model.encode(texts, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Ошибка индексации {filename}: {e}")

    def _file_hash(self, file_path: str) -> str:
        """Вычисление хеша файла"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def auto_update(self) -> None:
        """Автоматическое обновление баз данных"""
        while True:
            time.sleep(300)  # Проверка каждые 5 минут
            
            with self.lock:
                need_reload = False
                for db_name, db_info in self.databases.items():
                    current_hash = self._file_hash(db_info['path'])
                    if current_hash != db_info['hash']:
                        need_reload = True
                        break
                        
            if need_reload:
                logger.info("Обнаружены изменения в базах данных. Перезагрузка...")
                self.load_all_databases()

    def search_in_text(self, content: str, query: str, file_name: str) -> List[Dict]:
        """Поиск в текстовом файле"""
        results = []
        query_lower = query.lower()
        
        for line in content.split('\n'):
            if query_lower in line.lower():
                results.append({
                    'db': file_name,
                    'score': 1.0,
                    'data': {'line': line.strip()}
                })
                if len(results) >= CONFIG.max_results:
                    break
        
        return results

    def _search_dataframe(self, df: pd.DataFrame, query: str, db_name: str) -> List[Dict]:
        """Поиск в DataFrame"""
        results = []
        query_lower = query.lower()
        text_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
        
        if text_cols:
            mask = df[text_cols].apply(
                lambda col: col.astype(str).str.lower().str.contains(query_lower, regex=False, na=False)
            ).any(axis=1)
            
            for _, row in df[mask].iterrows():
                results.append({
                    'db': db_name,
                    'score': 0.8,
                    'data': row.to_dict()
                })
                if len(results) >= CONFIG.max_results:
                    break
                
        return results

    def _search_json(self, data: Union[Dict, List], query: str, path: str = "") -> List[Dict]:
        """Рекурсивный поиск в JSON"""
        results = []
        query_lower = query.lower()
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                results.extend(self._search_json(value, query, current_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                results.extend(self._search_json(item, query, f"{path}[{i}]"))
        elif isinstance(data, str) and query_lower in data.lower():
            results.append({path: data})
            
        return results

    def _semantic_search(self, query: str) -> List[Dict]:
        """Семантический поиск"""
        if not model_loaded or not model:
            return []
            
        try:
            query_embedding = model.encode([query], show_progress_bar=False)
            results = []
            
            for filename, emb in self.embeddings.items():
                similarities = cosine_similarity(query_embedding, emb)[0]
                top_indices = np.where(similarities >= CONFIG.similarity_threshold)[0]
                
                if len(top_indices) > 0:
                    df = self.databases[filename]['data']
                    for idx in top_indices[:CONFIG.max_results]:
                        results.append({
                            'db': filename,
                            'score': float(similarities[idx]),
                            'data': df.iloc[idx].to_dict()
                        })
                        if len(results) >= CONFIG.max_results:
                            break
            
            return results
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            return []

    def _process_results(self, results: List[Dict]) -> List[Dict]:
        """Обработка результатов"""
        seen = set()
        unique_results = []
        
        for res in results:
            data_str = str(res.get('data', ''))
            if data_str not in seen:
                seen.add(data_str)
                unique_results.append(res)
                if len(unique_results) >= CONFIG.max_results:
                    break
        
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        return unique_results

    def search(self, query: str, search_type: str = "universal") -> List[Dict]:
        """Унифицированный поиск"""
        query = str(query).strip()
        if not query:
            return []
        
        results = []
        
        with self.lock:
            for filename, db_info in self.databases.items():
                if db_info['type'] == 'text':
                    results.extend(self.search_in_text(db_info['content'], query, filename))
                elif db_info['type'] == 'table':
                    results.extend(self._search_dataframe(db_info['data'], query, filename))
                else:
                    json_results = self._search_json(db_info['data'], query)
                    if json_results:
                        results.append({
                            'db': filename,
                            'score': 0.5,
                            'data': {"matches": json_results}
                        })
                
                if len(results) >= CONFIG.max_results:
                    break
        
        if model_loaded and query.strip():
            semantic_results = self._semantic_search(query)
            results.extend(semantic_results)
            if len(results) >= CONFIG.max_results * 2:
                results = results[:CONFIG.max_results * 2]
        
        return self._process_results(results)

searcher = DataSearcher(CONFIG.data_folder)

# ==================== УТИЛИТЫ ====================
def create_keyboard(buttons: List[List[Dict]]) -> types.InlineKeyboardMarkup:
    """Создание inline-клавиатуры (оптимизировано)"""
    markup = types.InlineKeyboardMarkup()
    for row in buttons:
        keyboard_row = []
        for btn in row:
            if isinstance(btn, dict) and 'text' in btn and 'callback_data' in btn:
                keyboard_row.append(types.InlineKeyboardButton(
                    text=str(btn['text']),
                    callback_data=str(btn['callback_data'])
                ))
        if keyboard_row:
            markup.row(*keyboard_row)
    return markup

def show_typing(chat_id: int, duration: float = 0.5) -> None:
    """Отображение индикатора набора текста"""
    try:
        bot.send_chat_action(chat_id, 'typing')
        time.sleep(duration)
    except Exception as e:
        logger.error(f"Ошибка в show_typing: {e}")

def format_results(results: List[Dict], query: str) -> str:
    """Форматирование результатов поиска с учетом лимита Telegram"""
    if not results:
        return "❌ Ничего не найдено по вашему запросу."
    
    response = [
        f"🔎 <b>Результаты поиска:</b> {query}",
        f"📊 <b>Найдено совпадений:</b> {len(results)}",
        f"📂 <b>Базы данных:</b> {', '.join(set(r['db'] for r in results))}",
        "",
        "<b>Топ результатов:</b>"
    ]
    
    for i, res in enumerate(results[:5], 1):
        if 'line' in res['data']:
            response.append(f"\n📝 <b>Результат {i}</b> (из {res['db']})")
            response.append(f"<code>{res['data']['line']}</code>")
        else:
            response.append(f"\n📌 <b>Результат {i}</b> (релевантность: {res['score']:.2f})")
            for key, value in res['data'].items():
                response.append(f"  • <b>{key}:</b> {value}")
    
    # Обрезаем, если превышает лимит Telegram
    full_text = "\n".join(response)
    max_len = 4096
    if len(full_text) > max_len:
        # Обрезаем до ближайшего \n перед лимитом
        cut_text = full_text[:max_len-100]
        last_nl = cut_text.rfind('\n')
        if last_nl == -1:
            last_nl = max_len-100
        cut_text = cut_text[:last_nl]
        cut_text += "\n\n⚠️ Показана только часть результатов. Для полного отчета скачайте файл."
        return cut_text
    return full_text

def generate_report(results: List[Dict], query: str) -> Tuple[str, BytesIO]:
    """Генерация отчета в формате CSV"""
    try:
        if not results:
            return None, None
            
        # Создаем DataFrame из результатов
        flat_results = []
        for res in results:
            if 'line' in res['data']:
                flat_results.append({
                    'База данных': res['db'],
                    'Релевантность': res['score'],
                    'Результат': res['data']['line']
                })
            else:
                for key, value in res['data'].items():
                    flat_results.append({
                        'База данных': res['db'],
                        'Релевантность': res['score'],
                        'Поле': key,
                        'Значение': str(value)
                    })
        
        df = pd.DataFrame(flat_results)
        
        # Создаем CSV в памяти
        output = BytesIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        # Формируем имя файла
        filename = f"report_{datetime.now(CONFIG.timezone).strftime('%Y%m%d_%H%M%S')}.csv"
        
        return filename, output
    except Exception as e:
        logger.error(f"Ошибка генерации отчета: {e}")
        return None, None

def get_model_status() -> str:
    """Получение статуса модели"""
    if model_loaded:
        return "✅ Модель загружена и готова к работе"
    elif model_loading:
        elapsed = time.time() - model_load_start
        return f"🔄 Модель загружается... ({elapsed:.1f} сек)"
    else:
        return "❌ Модель не загружена (семантический поиск недоступен)"

def is_admin(user_id: int) -> bool:
    """Проверка, является ли пользователь администратором"""
    return user_id in CONFIG.admin_id

def format_user_info(user: Dict) -> str:
    """Форматирование информации о пользователе"""
    status = "👑 Админ" if user['user_id'] in CONFIG.admin_id else \
             "⭐ VIP" if user['is_vip'] else \
             "👤 Обычный"
    
    vip_info = ""
    if user['is_vip'] and user.get('vip_expiry_date'):
        expiry_date = datetime.strptime(user['vip_expiry_date'], '%Y-%m-%d')
        days_left = (expiry_date - datetime.now(CONFIG.timezone)).days
        vip_info = f"\nVIP истекает через: {days_left} дней"
    
    return (
        f"🆔 ID: {user['user_id']}\n"
        f"👤 Имя: {user['first_name']} {user['last_name'] or ''}\n"
        f"📅 Регистрация: {user['join_date']}\n"
        f"🔹 Статус: {status}{vip_info}\n"
        f"📊 Запросов сегодня: {user['requests_today']}/"
        f"{CONFIG.daily_limit['vip' if user['is_vip'] or user['user_id'] in CONFIG.vip_users else 'free']}\n"
        f"💰 Баланс: {user['balance']} руб."
    )

def format_ticket_info(ticket: Dict) -> str:
    """Форматирование информации о тикете"""
    user = user_db.get_user(ticket['user_id'])
    username = f"@{user['username']}" if user and user.get('username') else "нет username"
    
    status_map = {
        'open': '🟢 Открыт',
        'closed': '🔴 Закрыт',
        'pending': '🟡 Ожидает ответа'
    }
    
    admin_info = ""
    if ticket.get('admin_id'):
        admin = user_db.get_user(ticket['admin_id'])
        if admin:
            admin_info = f"\n👨‍💻 Администратор: {admin['first_name']} {admin['last_name'] or ''}"
    
    return (
        f"📌 <b>Тикет #{ticket['ticket_id']}</b>\n"
        f"📝 Тема: {ticket['subject']}\n"
        f"👤 Пользователь: {user['first_name']} {user['last_name'] or ''} ({username})\n"
        f"🆔 ID: {ticket['user_id']}\n"
        f"📅 Создан: {ticket['created_at']}\n"
        f"🔄 Обновлен: {ticket['updated_at']}\n"
        f"🔹 Статус: {status_map.get(ticket['status'], ticket['status'])}{admin_info}"
    )

def format_ticket_message(msg: Dict) -> str:
    """Форматирование сообщения тикета"""
    user = user_db.get_user(msg['user_id'])
    username = f"@{user['username']}" if user and user.get('username') else "нет username"
    
    sender = "👨‍💻 Админ" if msg['is_admin'] else f"👤 {user['first_name']} {user['last_name'] or ''} ({username})"
    
    return (
        f"<b>{sender}</b> [{msg['created_at']}]:\n"
        f"{msg['message']}\n"
    )

# ==================== ОБРАБОТЧИКИ КОМАНД ====================
@bot.message_handler(commands=['start', 'menu'])
def handle_start(message: types.Message) -> None:
    if not is_actual_message(message):
        return
    try:
        # Добавляем/обновляем пользователя в базе данных
        user_db.add_user(message.from_user)
        
        # Получаем информацию о пользователе
        user = user_db.get_user(message.from_user.id)
        
        # Статус модели для отображения
        model_status = get_model_status()
        
        # Создаем клавиатуру меню
        buttons = [
            [{'text': '🔍 Поиск информации', 'callback_data': 'search_menu'}],
            [{'text': '👤 Мой профиль', 'callback_data': 'profile'},
             {'text': 'ℹ️ Помощь', 'callback_data': 'help'}],
            [{'text': '🆘 Поддержка', 'callback_data': 'support_menu'}]
        ]
        
        # Добавляем кнопку админ-панели для администраторов
        if is_admin(message.from_user.id):
            buttons.append([{'text': '👑 Админ-панель', 'callback_data': 'admin_panel'}])
        
        # Показываем индикатор набора сообщения
        show_typing(message.chat.id)
        
        # Отправляем приветственное сообщение
        bot.send_message(
            chat_id=message.chat.id,
            text=f"👁️ <b>Глаза Бога</b> - система поиска информации\n\n"
                 f"Привет, {message.from_user.first_name}!\n\n"
                 f"<b>Статус системы:</b>\n{model_status}\n\n"
                 "Выберите действие из меню ниже:",
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        
    except Exception as e:
        logger.error(f"Ошибка в handle_start: {e}")
        traceback.print_exc()
        
        # Отправляем сообщение об ошибке
        try:
            bot.reply_to(
                message,
                "⚠️ Произошла ошибка при обработке команды. Пожалуйста, попробуйте позже."
            )
        except Exception as send_error:
            logger.error(f"Ошибка при отправке сообщения об ошибке: {send_error}")

@bot.callback_query_handler(func=lambda call: call.data == 'search_menu')
def show_search_menu(call: types.CallbackQuery) -> None:
    """Меню поиска"""
    buttons = [
        [{'text': '🔍 Универсальный поиск', 'callback_data': 'universal_search'}],
        [{'text': '📞 По номеру телефона', 'callback_data': 'search_phone'}],
        [{'text': '📧 По email', 'callback_data': 'search_email'}],
        [{'text': '🌐 По IP/домену', 'callback_data': 'search_ip'}],
        [{'text': '🔙 Назад', 'callback_data': 'main_menu'}]
    ]
    show_typing(call.message.chat.id)
    try:
        bot.edit_message_text(
            "🔍 <b>Меню поиска</b>\n\nВыберите тип поиска:",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
    except Exception:
        bot.send_message(
            call.message.chat.id,
            "🔍 <b>Меню поиска</b>\n\nВыберите тип поиска:",
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data in ['universal_search', 'search_phone', 'search_email', 'search_ip'])
def handle_search_type(call: types.CallbackQuery) -> None:
    """Обработка выбора типа поиска"""
    search_types = {
        'universal_search': ('любые данные', 'universal', "🔍 Можно ввести любой текст для поиска"),
        'search_phone': ('номер телефона', 'phone', "📞 +79161234567\n📞 89161234567"),
        'search_email': ('email', 'email', "📧 example@mail.ru\n📧 test@gmail.com"),
        'search_ip': ('IP/домен', 'ip', "🌐 192.168.1.1\n🌐 google.com")
    }
    
    search_info = search_types[call.data]
    
    try:
        msg = bot.send_message(
            call.message.chat.id,
            f"🔍 Введите {search_info[0]} для поиска:\n\n"
            f"<b>Примеры:</b>\n{search_info[2]}\n\n"
            f"<i>Поддерживаются различные форматы</i>",
            parse_mode='HTML'
        )
        bot.register_next_step_handler(msg, lambda m: perform_search(m, search_info[1]))
        safe_answer_callback_query(call)
    except Exception as e:
        logger.error(f"Ошибка в handle_search_type: {e}")
        safe_answer_callback_query(call, "⚠️ Ошибка. Попробуйте позже.", show_alert=True)

def perform_search(message: types.Message, search_type: str) -> None:
    """Выполнение поиска"""
    try:
        user_id = message.from_user.id
        if not model_loaded:
            bot.reply_to(message, "❌ Модель еще не загружена. Попробуйте позже.")
            return
        if not user_db.check_request_limit(user_id):
            show_typing(message.chat.id)
            bot.send_message(
                message.chat.id,
                "⚠️ <b>Лимит запросов исчерпан!</b>\n\n"
                "Вы использовали все доступные запросы на сегодня.\n"
                f"Лимит обновится через: <b>{(24 - datetime.now(CONFIG.timezone).hour) - 1} часов</b>\n\n"
                "🆙 Хотите увеличить лимит? Получите VIP-статус!",
                parse_mode='HTML'
            )
            return
        query = message.text.strip()
        if not query:
            bot.reply_to(message, "❌ Запрос не может быть пустым.")
            return
        search_msg = bot.send_message(message.chat.id, "🔍 <i>Идет поиск информации...</i>", parse_mode='HTML')
        bot.send_chat_action(message.chat.id, 'typing')
        time.sleep(1)
        results = searcher.search(query, search_type)
        user_db.update_request_stats(user_id, query, len(results), search_type)
        bot.delete_message(search_msg.chat.id, search_msg.message_id)
        buttons = [
            [{'text': '📥 Скачать отчет', 'callback_data': f'download_{query}'}],
            [{'text': '🔍 Новый поиск', 'callback_data': 'search_menu'},
             {'text': '📊 Профиль', 'callback_data': 'profile'}]
        ]
        show_typing(message.chat.id, 1)
        result_text = format_results(results, query)
        if len(result_text) > 4096:
            result_text = result_text[:4090] + "..."
        bot.send_message(
            message.chat.id,
            result_text,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Ошибка в perform_search: {e}")
        traceback.print_exc()
        try:
            bot.reply_to(message, "⚠️ Произошла ошибка при поиске. Попробуйте позже.")
        except Exception as send_error:
            logger.error(f"Ошибка при отправке сообщения об ошибке: {send_error}")

@bot.callback_query_handler(func=lambda call: call.data == 'profile')
def show_profile(call: types.CallbackQuery) -> None:
    """Отображение профиля пользователя"""
    try:
        user = user_db.get_user(call.from_user.id)
        if not user:
            bot.answer_callback_query(call.id, "❌ Пользователь не найден")
            return
            
        history = user_db.get_search_history(user['user_id'])
        history_text = "\n".join(
            f"{i+1}. {item['query']} ({item['results_count']}) - {item['date']}"
            for i, item in enumerate(history)
        ) if history else "Нет истории поиска"
        
        buttons = [
            [{'text': '🆙 VIP статус', 'callback_data': 'vip_upgrade'}],
            [{'text': '📜 Вся история', 'callback_data': 'full_history'},
             {'text': '🔄 Обновить', 'callback_data': 'profile'}],
            [{'text': '🔙 Назад', 'callback_data': 'main_menu'}]
        ]
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            f"👤 <b>Ваш профиль</b>\n\n"
            f"{format_user_info(user)}\n\n"
            f"🔎 <b>Последние запросы:</b>\n{history_text}",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в show_profile: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data == 'vip_upgrade')
def vip_upgrade(call: types.CallbackQuery) -> None:
    """Обновление VIP статуса"""
    try:
        buttons = [
            [{'text': f'💳 Купить VIP ({CONFIG.vip_price} руб/мес)', 'callback_data': 'buy_vip'}],
            [{'text': '🔙 Назад', 'callback_data': 'profile'}]
        ]
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            "🆙 <b>VIP Статус</b>\n\n"
            "Преимущества VIP статуса:\n"
            "• Увеличенный лимит запросов (15/день)\n"
            "• Приоритетная поддержка\n"
            "• Доступ к новым функциям первым\n\n"
            f"Стоимость: {CONFIG.vip_price} руб./месяц",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в vip_upgrade: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data == 'buy_vip')
def buy_vip(call: types.CallbackQuery) -> None:
    """Покупка VIP статуса"""
    try:
        user_id = call.from_user.id
        with user_db.lock:
            user_db.cursor.execute(
                "UPDATE users SET is_vip = TRUE, vip_expiry_date = date('now', '+1 month') WHERE user_id = ?",
                (user_id,)
            )
            user_db.conn.commit()
        
        bot.answer_callback_query(
            call.id,
            f"✅ VIP статус успешно активирован на 1 месяц!",
            show_alert=True
        )
        show_profile(call)
    except Exception as e:
        logger.error(f"Ошибка в buy_vip: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка при покупке VIP")

@bot.callback_query_handler(func=lambda call: call.data == 'full_history')
def show_full_history(call: types.CallbackQuery) -> None:
    """Показ полной истории поиска"""
    try:
        history = user_db.get_search_history(call.from_user.id, limit=20)
        history_text = "\n".join(
            f"{i+1}. {item['query']} ({item['results_count']}) - {item['date']}"
            for i, item in enumerate(history)
        ) if history else "Нет истории поиска"
        
        buttons = [
            [{'text': '🔙 Назад', 'callback_data': 'profile'}]
        ]
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            f"📜 <b>Полная история поиска</b>\n\n{history_text}",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в show_full_history: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data == 'help')
def show_help(call: types.CallbackQuery) -> None:
    """Показ помощи"""
    try:
        help_text = (
            "🆘 <b>Помощь по боту</b>\n\n"
            "🔍 <b>Поиск информации:</b>\n"
            "- Используйте меню поиска для выбора типа данных\n"
            "- Введите запрос для поиска\n\n"
            "👤 <b>Профиль:</b>\n"
            "- Просмотр статистики и истории запросов\n\n"
            "🆙 <b>VIP статус:</b>\n"
            "- Увеличивает лимит запросов\n\n"
            "🆘 <b>Поддержка:</b>\n"
            "- Используйте меню поддержки для создания тикетов\n"
            "- Администраторы ответят вам в кратчайшие сроки\n\n"
            "📢 <b>Поддержка:</b>\n"
            "@QVENTIS_TEAM"
        )
        
        buttons = [
            [{'text': '🔙 Назад', 'callback_data': 'main_menu'}]
        ]
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            help_text,
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в show_help: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

# ==================== ПОДДЕРЖКА ====================
@bot.callback_query_handler(func=lambda call: call.data == 'support_menu')
def show_support_menu(call: types.CallbackQuery) -> None:
    """Меню поддержки"""
    try:
        user_id = call.from_user.id
        tickets = user_db.get_user_tickets(user_id)
        
        buttons = [
            [{'text': '🆕 Создать тикет', 'callback_data': 'create_ticket'}],
            [{'text': '📜 Мои тикеты', 'callback_data': 'my_tickets'}],
            [{'text': '🔙 Назад', 'callback_data': 'main_menu'}]
        ]
        
        if is_admin(user_id):
            buttons.insert(1, [{'text': '👨‍💻 Тикеты пользователей', 'callback_data': 'admin_tickets'}])
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            "🆘 <b>Поддержка</b>\n\n"
            "Здесь вы можете создать тикет для связи с администрацией.\n"
            f"Максимальное количество открытых тикетов: {CONFIG.max_tickets_per_user}\n\n"
            f"Ваши текущие тикеты: {len([t for t in tickets if t['status'] == 'open'])}/{CONFIG.max_tickets_per_user}",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в show_support_menu: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data == 'create_ticket')
def create_ticket(call: types.CallbackQuery) -> None:
    """Создание тикета"""
    try:
        user_id = call.from_user.id
        tickets = user_db.get_user_tickets(user_id)
        open_tickets = len([t for t in tickets if t['status'] == 'open'])
        
        if open_tickets >= CONFIG.max_tickets_per_user:
            bot.answer_callback_query(
                call.id,
                f"❌ У вас уже есть {open_tickets} открытых тикетов. Максимум: {CONFIG.max_tickets_per_user}",
                show_alert=True
            )
            return
        
        msg = bot.send_message(
            call.message.chat.id,
            "📝 Введите тему вашего обращения (или 'отмена' для отмены):",
            parse_mode='HTML'
        )
        bot.register_next_step_handler(msg, process_ticket_subject)
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в create_ticket: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

def process_ticket_subject(message: types.Message) -> None:
    """Обработка темы тикета"""
    try:
        if message.text.lower() == 'отмена':
            bot.send_message(message.chat.id, "❌ Создание тикета отменено")
            return
        
        if len(message.text) > 100:
            msg = bot.send_message(message.chat.id, "❌ Тема слишком длинная (макс. 100 символов). Введите снова:")
            bot.register_next_step_handler(msg, process_ticket_subject)
            return
        
        user_data = {'subject': message.text}
        
        msg = bot.send_message(
            message.chat.id,
            "📝 Теперь введите ваше сообщение для поддержки:",
            parse_mode='HTML'
        )
        bot.register_next_step_handler(msg, process_ticket_message, user_data)
    except Exception as e:
        logger.error(f"Ошибка в process_ticket_subject: {e}")
        bot.send_message(message.chat.id, "⚠️ Произошла ошибка. Попробуйте позже.")

def process_ticket_message(message: types.Message, user_data: Dict) -> None:
    """Обработка сообщения тикета"""
    try:
        if message.text.lower() == 'отмена':
            bot.send_message(message.chat.id, "❌ Создание тикета отменено")
            return
        
        ticket_id = user_db.create_support_ticket(
            message.from_user.id,
            user_data['subject'],
            message.text
        )
        
        if not ticket_id:
            bot.send_message(
                message.chat.id,
                f"❌ У вас уже есть максимальное количество открытых тикетов ({CONFIG.max_tickets_per_user})",
                parse_mode='HTML'
            )
            return
        
        # Уведомление администраторам
        for admin_id in CONFIG.admin_id:
            try:
                ticket = user_db.get_ticket(ticket_id)
                if ticket:
                    buttons = [
                        [{'text': '💬 Ответить', 'callback_data': f'reply_ticket_{ticket_id}'}],
                        [{'text': '🔒 Закрыть', 'callback_data': f'close_ticket_{ticket_id}'}]
                    ]
                    
                    bot.send_message(
                        admin_id,
                        f"🆘 <b>Новый тикет #{ticket_id}</b>\n\n"
                        f"{format_ticket_info(ticket)}\n\n"
                        f"📝 Сообщение:\n{message.text}",
                        reply_markup=create_keyboard(buttons),
                        parse_mode='HTML'
                    )
            except Exception as e:
                logger.error(f"Ошибка отправки уведомления администратору {admin_id}: {e}")
        
        buttons = [
            [{'text': '📜 Мои тикеты', 'callback_data': 'my_tickets'}],
            [{'text': '🔙 В меню', 'callback_data': 'main_menu'}]
        ]
        
        bot.send_message(
            message.chat.id,
            f"✅ <b>Тикет #{ticket_id} создан!</b>\n\n"
            "Администратор ответит вам в ближайшее время.\n"
            "Вы можете просмотреть статус тикета в разделе 'Мои тикеты'.",
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Ошибка в process_ticket_message: {e}")
        bot.send_message(message.chat.id, "⚠️ Произошла ошибка при создании тикета")

@bot.callback_query_handler(func=lambda call: call.data == 'my_tickets')
def show_user_tickets(call: types.CallbackQuery) -> None:
    """Показ тикетов пользователя (оптимизировано)"""
    try:
        tickets = user_db.get_user_tickets(call.from_user.id)
        if not tickets:
            safe_answer_callback_query(call, "У вас нет тикетов", show_alert=True)
            return
        buttons = []
        for ticket in tickets[:10]:  # Ограничиваем показ 10 тикетами
            status_icon = '🟢' if ticket['status'] == 'open' else '🔴'
            ticket_id = ticket['ticket_id']
            subject = ticket['subject']
            buttons.append([
                {'text': f"{status_icon} #{ticket_id} - {subject}",
                 'callback_data': f'view_ticket_{ticket_id}'}
            ])
        buttons.append([{'text': '🔙 Назад', 'callback_data': 'support_menu'}])
        show_typing(call.message.chat.id)
        try:
            bot.edit_message_text(
                "📜 <b>Ваши тикеты</b>\n\n"
                "Выберите тикет для просмотра:",
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                reply_markup=create_keyboard(buttons),
                parse_mode='HTML'
            )
        except Exception:
            bot.send_message(
                call.message.chat.id,
                "📜 <b>Ваши тикеты</b>\n\nВыберите тикет для просмотра:",
                reply_markup=create_keyboard(buttons),
                parse_mode='HTML'
            )
        safe_answer_callback_query(call)
    except Exception as e:
        logger.error(f"Ошибка в show_user_tickets: {e}")
        safe_answer_callback_query(call, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data.startswith('view_ticket_'))
def view_ticket(call: types.CallbackQuery) -> None:
    """Просмотр тикета"""
    try:
        ticket_id = call.data.split('_')[-1]
        ticket = user_db.get_ticket(ticket_id)
        
        if not ticket or ticket['user_id'] != call.from_user.id and not is_admin(call.from_user.id):
            bot.answer_callback_query(call.id, "❌ Тикет не найден или нет доступа", show_alert=True)
            return
        
        messages = user_db.get_ticket_messages(ticket_id)
        messages_text = "\n".join([format_ticket_message(msg) for msg in messages])
        
        buttons = []
        if is_admin(call.from_user.id):
            if ticket['status'] == 'open':
                buttons.append([{'text': '💬 Ответить', 'callback_data': f'reply_ticket_{ticket_id}'}])
                buttons.append([{'text': '🔒 Закрыть', 'callback_data': f'close_ticket_{ticket_id}'}])
            else:
                buttons.append([{'text': '💬 Ответить', 'callback_data': f'reply_ticket_{ticket_id}'}])
                buttons.append([{'text': '🟢 Открыть', 'callback_data': f'open_ticket_{ticket_id}'}])
        else:
            if ticket['status'] == 'open':
                buttons.append([{'text': '💬 Добавить сообщение', 'callback_data': f'add_ticket_msg_{ticket_id}'}])
        
        buttons.append([{'text': '🔙 Назад', 'callback_data': 'my_tickets' if not is_admin(call.from_user.id) else 'admin_tickets'}])
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            f"{format_ticket_info(ticket)}\n\n"
            f"📝 <b>Сообщения:</b>\n\n{messages_text}",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в view_ticket: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data.startswith('reply_ticket_', 'add_ticket_msg_'))
def reply_to_ticket(call: types.CallbackQuery) -> None:
    """Ответ на тикет"""
    try:
        ticket_id = call.data.split('_')[-1]
        ticket = user_db.get_ticket(ticket_id)
        
        if not ticket or (ticket['user_id'] != call.from_user.id and not is_admin(call.from_user.id)):
            safe_answer_callback_query(call, "❌ Тикет не найден или нет доступа", show_alert=True)
            return
        
        if ticket['status'] != 'open' and not is_admin(call.from_user.id):
            safe_answer_callback_query(call, "❌ Тикет закрыт. Новые сообщения добавлять нельзя", show_alert=True)
            return
        
        msg = bot.send_message(
            call.message.chat.id,
            "📝 Введите ваше сообщение (или 'отмена' для отмены):",
            parse_mode='HTML'
        )
        
        if call.data.startswith('reply_ticket_'):
            bot.register_next_step_handler(msg, process_ticket_reply, {'ticket_id': ticket_id, 'is_admin': is_admin(call.from_user.id)})
        else:
            bot.register_next_step_handler(msg, process_ticket_reply, {'ticket_id': ticket_id, 'is_admin': False})
        
        safe_answer_callback_query(call)
    except Exception as e:
        logger.error(f"Ошибка в reply_to_ticket: {e}")
        safe_answer_callback_query(call, "⚠️ Ошибка. Попробуйте позже.", show_alert=True)

def process_ticket_reply(message: types.Message, data: Dict) -> None:
    """Обработка ответа на тикет"""
    try:
        if message.text.lower() == 'отмена':
            bot.send_message(message.chat.id, "❌ Добавление сообщения отменено")
            return
        
        ticket_id = data['ticket_id']
        is_admin = data['is_admin']
        
        if not user_db.add_support_message(ticket_id, message.from_user.id, message.text, is_admin):
            raise Exception("Ошибка добавления сообщения в БД")
        
        ticket = user_db.get_ticket(ticket_id)
        if not ticket:
            raise Exception("Тикет не найден")
        
        # Уведомление другой стороны
        if is_admin:
            # Админ ответил - уведомляем пользователя
            try:
                buttons = [
                    [{'text': '💬 Ответить', 'callback_data': f'add_ticket_msg_{ticket_id}'}],
                    [{'text': '📜 Мои тикеты', 'callback_data': 'my_tickets'}]
                ]
                
                bot.send_message(
                    ticket['user_id'],
                    f"💬 <b>Новый ответ по тикету #{ticket_id}</b>\n\n"
                    f"👨‍💻 Администратор ответил на ваш тикет:\n"
                    f"{message.text}\n\n"
                    f"Вы можете ответить, нажав кнопку ниже.",
                    reply_markup=create_keyboard(buttons),
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"Ошибка отправки уведомления пользователю {ticket['user_id']}: {e}")
        else:
            # Пользователь ответил - уведомляем администраторов
            for admin_id in CONFIG.admin_id:
                try:
                    buttons = [
                        [{'text': '💬 Ответить', 'callback_data': f'reply_ticket_{ticket_id}'}],
                        [{'text': '🔒 Закрыть', 'callback_data': f'close_ticket_{ticket_id}'}]
                    ]
                    
                    bot.send_message(
                        admin_id,
                        f"💬 <b>Новый ответ по тикету #{ticket_id}</b>\n\n"
                        f"👤 Пользователь ответил на тикет:\n"
                        f"{message.text}",
                        reply_markup=create_keyboard(buttons),
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.error(f"Ошибка отправки уведомления администратору {admin_id}: {e}")
        
        bot.send_message(
            message.chat.id,
            f"✅ <b>Ваше сообщение добавлено в тикет #{ticket_id}</b>",
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Ошибка в process_ticket_reply: {e}")
        bot.send_message(message.chat.id, "⚠️ Произошла ошибка при добавлении сообщения")

@bot.callback_query_handler(func=lambda call: call.data.startswith(('close_ticket_', 'open_ticket_')))
def change_ticket_status(call: types.CallbackQuery) -> None:
    """Изменение статуса тикета"""
    try:
        if not is_admin(call.from_user.id):
            bot.answer_callback_query(call.id, "⛔ Доступ запрещен", show_alert=True)
            return
        
        action, ticket_id = call.data.split('_')[:2]
        ticket_id = call.data.split('_')[-1]
        new_status = 'closed' if action == 'close' else 'open'
        
        if not user_db.update_ticket_status(ticket_id, new_status, call.from_user.id):
            raise Exception("Ошибка обновления статуса тикета")
        
        ticket = user_db.get_ticket(ticket_id)
        if not ticket:
            raise Exception("Тикет не найден")
        
        # Уведомление пользователя
        if new_status == 'closed':
            try:
                bot.send_message(
                    ticket['user_id'],
                    f"🔒 <b>Ваш тикет #{ticket_id} закрыт администратором</b>\n\n"
                    f"Если у вас остались вопросы, вы можете создать новый тикет.",
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"Ошибка отправки уведомления пользователю {ticket['user_id']}: {e}")
        else:
            try:
                buttons = [
                    [{'text': '💬 Ответить', 'callback_data': f'add_ticket_msg_{ticket_id}'}],
                    [{'text': '📜 Мои тикеты', 'callback_data': 'my_tickets'}]
                ]
                
                bot.send_message(
                    ticket['user_id'],
                    f"🟢 <b>Ваш тикет #{ticket_id} снова открыт администратором</b>\n\n"
                    f"Вы можете продолжить общение.",
                    reply_markup=create_keyboard(buttons),
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"Ошибка отправки уведомления пользователю {ticket['user_id']}: {e}")
        
        bot.answer_callback_query(
            call.id,
            f"✅ Тикет #{ticket_id} {'закрыт' if new_status == 'closed' else 'открыт'}",
            show_alert=True
        )
        
        # Обновляем просмотр тикета
        view_ticket(call)
    except Exception as e:
        logger.error(f"Ошибка в change_ticket_status: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка при изменении статуса тикета")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_tickets')
def show_admin_tickets(call: types.CallbackQuery) -> None:
    """Показ тикетов для администратора (оптимизировано)"""
    try:
        if not is_admin(call.from_user.id):
            safe_answer_callback_query(call, "⛔ Доступ запрещен", show_alert=True)
            return
        tickets = user_db.get_open_tickets()
        if not tickets:
            safe_answer_callback_query(call, "Нет открытых тикетов", show_alert=True)
            return
        buttons = []
        for ticket in tickets[:10]:  # Ограничиваем показ 10 тикетами
            user = user_db.get_user(ticket['user_id'])
            username = f"@{user['username']}" if user and user.get('username') else "нет username"
            ticket_id = ticket['ticket_id']
            first_name = user['first_name'] if user else ''
            buttons.append([
                {'text': f"🟢 #{ticket_id} - {first_name} ({username})",
                 'callback_data': f'view_ticket_{ticket_id}'}
            ])
        buttons.append([{'text': '🔙 Назад', 'callback_data': 'support_menu'}])
        show_typing(call.message.chat.id)
        try:
            bot.edit_message_text(
                "👨‍💻 <b>Открытые тикеты пользователей</b>\n\n"
                "Выберите тикет для просмотра:",
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                reply_markup=create_keyboard(buttons),
                parse_mode='HTML'
            )
        except Exception:
            bot.send_message(
                call.message.chat.id,
                "👨‍💻 <b>Открытые тикеты пользователей</b>\n\nВыберите тикет для просмотра:",
                reply_markup=create_keyboard(buttons),
                parse_mode='HTML'
            )
        safe_answer_callback_query(call)
    except Exception as e:
        logger.error(f"Ошибка в show_admin_tickets: {e}")
        safe_answer_callback_query(call, "⚠️ Ошибка. Попробуйте позже.")

# ==================== АДМИН ПАНЕЛЬ ====================
@bot.callback_query_handler(func=lambda call: call.data == 'admin_panel')
def show_admin_panel(call: types.CallbackQuery) -> None:
    """Панель администратора"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        buttons = [
            [{'text': '📊 Статистика', 'callback_data': 'admin_stats'}],
            [{'text': '🔄 Обновить базы', 'callback_data': 'admin_reload'},
             {'text': '📦 Бэкап данных', 'callback_data': 'admin_backup'}],
            [{'text': '📢 Рассылка', 'callback_data': 'admin_broadcast'}],
            [{'text': '👤 Управление VIP', 'callback_data': 'admin_vip'}],
            [{'text': '🚫 Управление блокировками', 'callback_data': 'admin_bans'}],
            [{'text': '🆘 Управление поддержкой', 'callback_data': 'admin_support'}],
            [{'text': '🔙 Назад', 'callback_data': 'main_menu'}]
        ]
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            "👑 <b>Административная панель</b>\n\n"
            "Выберите действие:",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в show_admin_panel: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

def safe_answer_callback_query(call: types.CallbackQuery, text: str = None, show_alert: bool = False) -> None:
    """Безопасная обработка callback-запросов с учетом их времени жизни (унифицировано)"""
    try:
        if text:
            bot.answer_callback_query(call.id, text, show_alert=show_alert)
        else:
            bot.answer_callback_query(call.id)
    except Exception as e:
        if "query is too old" in str(e) or "query ID is invalid" in str(e):
            logger.debug(f"Callback query expired or invalid: {e}")
        else:
            logger.error(f"Error answering callback query: {e}")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_stats')
def admin_stats(call: types.CallbackQuery) -> None:
    """Статистика администратора"""
    if not is_admin(call.from_user.id):
        safe_answer_callback_query(call, "⛔ Доступ запрещен", show_alert=True)
        return
    
    try:
        stats = user_db.get_full_stats()
        
        top_queries = "\n".join(
            f"{i+1}. {query[0]} ({query[1]})"
            for i, query in enumerate(stats.get('top_queries', []))
        )
        
        buttons = [
            [{'text': '🔄 Обновить', 'callback_data': 'admin_stats'}],
            [{'text': '🔙 Назад', 'callback_data': 'admin_panel'}]
        ]
        
        show_typing(call.message.chat.id)
        try:
            bot.edit_message_text(
                f"📊 <b>Статистика системы</b>\n\n"
                f"• Всего пользователей: {stats.get('total_users', 0)}\n"
                f"• VIP пользователей: {stats.get('vip_users', 0)}\n"
                f"• Всего запросов: {stats.get('total_searches', 0)}\n"
                f"• Всего тикетов: {stats.get('total_tickets', 0)}\n\n"
                f"🔝 <b>Популярные запросы:</b>\n{top_queries}",
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                reply_markup=create_keyboard(buttons),
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Error editing message: {e}")
            bot.send_message(
                call.message.chat.id,
                f"📊 <b>Статистика системы</b>\n\n"
                f"• Всего пользователей: {stats.get('total_users', 0)}\n"
                f"• VIP пользователей: {stats.get('vip_users', 0)}\n"
                f"• Всего запросов: {stats.get('total_searches', 0)}\n"
                f"• Всего тикетов: {stats.get('total_tickets', 0)}\n\n"
                f"🔝 <b>Популярные запросы:</b>\n{top_queries}",
                reply_markup=create_keyboard(buttons),
                parse_mode='HTML'
            )
    except Exception as e:
        logger.error(f"Ошибка в admin_stats: {e}")
        try:
            bot.send_message(
                call.message.chat.id,
                "⚠️ Произошла ошибка при получении статистики. Попробуйте позже.",
                parse_mode='HTML'
            )
        except Exception as send_error:
            logger.error(f"Error sending error message: {send_error}")
    finally:
        safe_answer_callback_query(call)

@bot.callback_query_handler(func=lambda call: call.data == 'admin_reload')
def admin_reload(call: types.CallbackQuery) -> None:
    """Обновление баз данных"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        searcher.load_all_databases()
        bot.answer_callback_query(call.id, "🔄 Базы данных успешно перезагружены")
    except Exception as e:
        logger.error(f"Ошибка в admin_reload: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка при перезагрузке")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_backup')
def admin_backup(call: types.CallbackQuery) -> None:
    """Создание бэкапа данных"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        # Создаем временный файл
        backup_file = BytesIO()
        with zipfile.ZipFile(backup_file, 'w') as zipf:
            # Добавляем базу данных
            if os.path.exists(user_db.db_path):
                zipf.write(user_db.db_path, os.path.basename(user_db.db_path))
            
            # Добавляем файлы данных
            for root, _, files in os.walk(CONFIG.data_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, CONFIG.data_folder))
        
        backup_file.seek(0)
        
        # Отправляем архив
        bot.send_document(
            call.message.chat.id,
            backup_file,
            caption=f"📦 <b>Бэкап данных</b>\n"
                   f"Дата: {datetime.now(CONFIG.timezone).strftime('%Y-%m-%d %H:%M:%S')}",
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в admin_backup: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка при создании бэкапа")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_broadcast')
def admin_broadcast(call: types.CallbackQuery) -> None:
    """Рассылка сообщений"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        msg = bot.send_message(
            call.message.chat.id,
            "📢 Введите сообщение для рассылки всем пользователям (или 'отмена' для отмены):",
            parse_mode='HTML'
        )
        bot.register_next_step_handler(msg, process_broadcast_message)
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в admin_broadcast: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

def process_broadcast_message(message: types.Message) -> None:
    """Обработка сообщения для рассылки"""
    if not is_admin(message.from_user.id):
        return
    
    try:
        if message.text.lower() == 'отмена':
            bot.send_message(message.chat.id, "❌ Рассылка отменена")
            return
        
        with user_db.lock:
            user_db.cursor.execute("SELECT user_id FROM users")
            user_ids = [row[0] for row in user_db.cursor.fetchall()]
        
        success = 0
        errors = 0
        total = len(user_ids)
        
        progress_msg = bot.send_message(
            message.chat.id,
            f"📢 Начата рассылка для {total} пользователей...\n\n"
            f"✅ Успешно: {success}\n"
            f"❌ Ошибок: {errors}",
            parse_mode='HTML'
        )
        
        for user_id in user_ids:
            try:
                bot.send_message(
                    user_id,
                    f"📢 <b>Важное сообщение от администратора:</b>\n\n{message.text}",
                    parse_mode='HTML'
                )
                success += 1
            except Exception as e:
                logger.warning(f"Не удалось отправить сообщение пользователю {user_id}: {e}")
                errors += 1
            
            # Обновляем статус каждые 10 отправок
            if (success + errors) % 10 == 0:
                try:
                    bot.edit_message_text(
                        f"📢 Рассылка для {total} пользователей...\n\n"
                        f"✅ Успешно: {success}\n"
                        f"❌ Ошибок: {errors}",
                        chat_id=progress_msg.chat.id,
                        message_id=progress_msg.message_id,
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.warning(f"Ошибка обновления статуса рассылки: {e}")
        
        try:
            bot.edit_message_text(
                f"✅ <b>Рассылка завершена</b>\n\n"
                f"• Всего пользователей: {total}\n"
                f"• Успешно отправлено: {success}\n"
                f"• Не удалось отправить: {errors}",
                chat_id=progress_msg.chat.id,
                message_id=progress_msg.message_id,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.warning(f"Ошибка финального сообщения рассылки: {e}")
    except Exception as e:
        logger.error(f"Ошибка в process_broadcast_message: {e}")
        try:
            bot.send_message(message.chat.id, "⚠️ Произошла ошибка при рассылке")
        except Exception as send_error:
            logger.error(f"Ошибка при отправке сообщения об ошибке: {send_error}")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_vip')
def admin_vip(call: types.CallbackQuery) -> None:
    """Управление VIP статусами"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        buttons = [
            [{'text': '➕ Добавить VIP', 'callback_data': 'admin_add_vip'},
             {'text': '➖ Удалить VIP', 'callback_data': 'admin_remove_vip'}],
            [{'text': '🔙 Назад', 'callback_data': 'admin_panel'}]
        ]
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            "👤 <b>Управление VIP статусами</b>\n\n"
            "Выберите действие:",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в admin_vip: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_add_vip')
def admin_add_vip(call: types.CallbackQuery) -> None:
    """Добавление VIP статуса"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        msg = bot.send_message(
            call.message.chat.id,
            "👤 Введите ID пользователя для добавления VIP статуса:",
            parse_mode='HTML'
        )
        bot.register_next_step_handler(msg, process_add_vip)
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в admin_add_vip: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

def process_add_vip(message: types.Message) -> None:
    """Обработка добавления VIP статуса"""
    if not is_admin(message.from_user.id):
        return
    
    try:
        user_id = int(message.text)
        if user_db.add_vip(user_id):
            bot.send_message(
                message.chat.id,
                f"✅ Пользователю {user_id} добавлен VIP статус на 1 месяц",
                parse_mode='HTML'
            )
        else:
            bot.send_message(
                message.chat.id,
                f"❌ Не удалось добавить VIP статус пользователю {user_id}",
                parse_mode='HTML'
            )
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неверный формат ID пользователя")
    except Exception as e:
        logger.error(f"Ошибка в process_add_vip: {e}")
        bot.send_message(message.chat.id, "⚠️ Произошла ошибка")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_remove_vip')
def admin_remove_vip(call: types.CallbackQuery) -> None:
    """Удаление VIP статуса"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        msg = bot.send_message(
            call.message.chat.id,
            "👤 Введите ID пользователя для удаления VIP статуса:",
            parse_mode='HTML'
        )
        bot.register_next_step_handler(msg, process_remove_vip)
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в admin_remove_vip: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

def process_remove_vip(message: types.Message) -> None:
    """Обработка удаления VIP статуса"""
    if not is_admin(message.from_user.id):
        return
    
    try:
        user_id = int(message.text)
        if user_db.remove_vip(user_id):
            bot.send_message(
                message.chat.id,
                f"✅ У пользователя {user_id} удален VIP статус",
                parse_mode='HTML'
            )
        else:
            bot.send_message(
                message.chat.id,
                f"❌ Не удалось удалить VIP статус пользователю {user_id}",
                parse_mode='HTML'
            )
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неверный формат ID пользователя")
    except Exception as e:
        logger.error(f"Ошибка в process_remove_vip: {e}")
        bot.send_message(message.chat.id, "⚠️ Произошла ошибка")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_bans')
def admin_bans(call: types.CallbackQuery) -> None:
    """Управление блокировками"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        buttons = [
            [{'text': '⛔ Заблокировать', 'callback_data': 'admin_ban'},
             {'text': '✅ Разблокировать', 'callback_data': 'admin_unban'}],
            [{'text': '🔙 Назад', 'callback_data': 'admin_panel'}]
        ]
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            "🚫 <b>Управление блокировками</b>\n\n"
            "Выберите действие:",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в admin_bans: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_ban')
def admin_ban(call: types.CallbackQuery) -> None:
    """Блокировка пользователя"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        msg = bot.send_message(
            call.message.chat.id,
            "👤 Введите ID пользователя для блокировки:",
            parse_mode='HTML'
        )
        bot.register_next_step_handler(msg, process_ban_user)
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в admin_ban: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

def process_ban_user(message: types.Message) -> None:
    """Обработка блокировки пользователя"""
    if not is_admin(message.from_user.id):
        return
    
    try:
        user_id = int(message.text)
        if user_db.ban_user(user_id):
            bot.send_message(
                message.chat.id,
                f"⛔ Пользователь {user_id} заблокирован",
                parse_mode='HTML'
            )
        else:
            bot.send_message(
                message.chat.id,
                f"❌ Не удалось заблокировать пользователя {user_id}",
                parse_mode='HTML'
            )
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неверный формат ID пользователя")
    except Exception as e:
        logger.error(f"Ошибка в process_ban_user: {e}")
        bot.send_message(message.chat.id, "⚠️ Произошла ошибка")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_unban')
def admin_unban(call: types.CallbackQuery) -> None:
    """Разблокировка пользователя"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        msg = bot.send_message(
            call.message.chat.id,
            "👤 Введите ID пользователя для разблокировки:",
            parse_mode='HTML'
        )
        bot.register_next_step_handler(msg, process_unban_user)
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в admin_unban: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

def process_unban_user(message: types.Message) -> None:
    """Обработка разблокировки пользователя"""
    if not is_admin(message.from_user.id):
        return
    
    try:
        user_id = int(message.text)
        if user_db.unban_user(user_id):
            bot.send_message(
                message.chat.id,
                f"✅ Пользователь {user_id} разблокирован",
                parse_mode='HTML'
            )
        else:
            bot.send_message(
                message.chat.id,
                f"❌ Не удалось разблокировать пользователя {user_id}",
                parse_mode='HTML'
            )
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неверный формат ID пользователя")
    except Exception as e:
        logger.error(f"Ошибка в process_unban_user: {e}")
        bot.send_message(message.chat.id, "⚠️ Произошла ошибка")

@bot.callback_query_handler(func=lambda call: call.data == 'admin_support')
def admin_support(call: types.CallbackQuery) -> None:
    """Управление поддержкой"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        buttons = [
            [{'text': '📜 Просмотреть тикеты', 'callback_data': 'admin_tickets'}],
            [{'text': '📊 Статистика поддержки', 'callback_data': 'support_stats'}],
            [{'text': '🔙 Назад', 'callback_data': 'admin_panel'}]
        ]
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            "🆘 <b>Управление поддержкой</b>\n\n"
            "Выберите действие:",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в admin_support: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data == 'support_stats')
def support_stats(call: types.CallbackQuery) -> None:
    """Статистика поддержки"""
    if not is_admin(call.from_user.id):
        bot.answer_callback_query(call.id, "⛔ Доступ запрещен")
        return
    
    try:
        with user_db.lock:
            user_db.cursor.execute("SELECT COUNT(*) FROM support_tickets")
            total_tickets = user_db.cursor.fetchone()[0]
            
            user_db.cursor.execute("SELECT COUNT(*) FROM support_tickets WHERE status = 'open'")
            open_tickets = user_db.cursor.fetchone()[0]
            
            user_db.cursor.execute("SELECT COUNT(*) FROM support_tickets WHERE status = 'closed'")
            closed_tickets = user_db.cursor.fetchone()[0]
            
            user_db.cursor.execute("""
                SELECT user_id, COUNT(*) as count 
                FROM support_tickets 
                GROUP BY user_id 
                ORDER BY count DESC 
                LIMIT 5
            """)
            top_users = user_db.cursor.fetchall()
        
        top_users_text = "\n".join(
            f"{i+1}. ID {user[0]} - {user[1]} тикетов"
            for i, user in enumerate(top_users)
        )
        
        buttons = [
            [{'text': '🔄 Обновить', 'callback_data': 'support_stats'}],
            [{'text': '🔙 Назад', 'callback_data': 'admin_support'}]
        ]
        
        show_typing(call.message.chat.id)
        bot.edit_message_text(
            f"📊 <b>Статистика поддержки</b>\n\n"
            f"• Всего тикетов: {total_tickets}\n"
            f"• Открытых тикетов: {open_tickets}\n"
            f"• Закрытых тикетов: {closed_tickets}\n\n"
            f"🔝 <b>Топ пользователей по тикетам:</b>\n{top_users_text}",
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=create_keyboard(buttons),
            parse_mode='HTML'
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Ошибка в support_stats: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка. Попробуйте позже.")

@bot.callback_query_handler(func=lambda call: call.data.startswith('download_'))
def download_report(call: types.CallbackQuery) -> None:
    """Скачивание отчета"""
    try:
        query = call.data.split('_', 1)[1]
        user_id = call.from_user.id
        
        # Проверяем историю пользователя
        history = user_db.get_search_history(user_id)
        found = False
        for item in history:
            if item['query'] == query:
                found = True
                break
        
        if not found:
            bot.answer_callback_query(call.id, "❌ Запрос не найден в вашей истории")
            return
        
        bot.answer_callback_query(call.id, "⏳ Отчет формируется...")
        
        # Имитируем формирование отчета
        time.sleep(1)
        
        # Генерируем отчет
        filename, report_file = generate_report(
            searcher.search(query),
            query
        )
        
        if not filename or not report_file:
            bot.answer_callback_query(call.id, "❌ Ошибка формирования отчета")
            return
        
        # Отправляем файл
        bot.send_document(
            call.message.chat.id,
            report_file,
            caption=f"📄 <b>Отчет по запросу:</b> {query}",
            visible_file_name=filename,
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Ошибка в download_report: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка при формировании отчета")

@bot.callback_query_handler(func=lambda call: call.data == 'main_menu')
def return_to_main_menu(call: types.CallbackQuery) -> None:
    """Возврат в главное меню"""
    try:
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text=f"👁️ <b>Глаза Бога</b> - система поиска информации\n\n"
                 f"Привет, {call.from_user.first_name}!\n\n"
                 f"<b>Статус системы:</b>\n{get_model_status()}\n\n"
                 "Выберите действие из меню ниже:",
            reply_markup=create_keyboard([
                [{'text': '🔍 Поиск информации', 'callback_data': 'search_menu'}],
                [{'text': '👤 Мой профиль', 'callback_data': 'profile'},
                 {'text': 'ℹ️ Помощь', 'callback_data': 'help'}],
                [{'text': '🆘 Поддержка', 'callback_data': 'support_menu'}]
            ] + ([ [{'text': '👑 Админ-панель', 'callback_data': 'admin_panel'}] ] if is_admin(call.from_user.id) else [])),
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Error editing message in return_to_main_menu: {e}")
        try:
            bot.send_message(
                chat_id=call.message.chat.id,
                text=f"👁️ <b>Глаза Бога</b> - система поиска информации\n\n"
                     f"Привет, {call.from_user.first_name}!\n\n"
                     f"<b>Статус системы:</b>\n{get_model_status()}\n\n"
                     "Выберите действие из меню ниже:",
                reply_markup=create_keyboard([
                    [{'text': '🔍 Поиск информации', 'callback_data': 'search_menu'}],
                    [{'text': '👤 Мой профиль', 'callback_data': 'profile'},
                     {'text': 'ℹ️ Помощь', 'callback_data': 'help'}],
                    [{'text': '🆘 Поддержка', 'callback_data': 'support_menu'}]
                ] + ([ [{'text': '👑 Админ-панель', 'callback_data': 'admin_panel'}] ] if is_admin(call.from_user.id) else [])),
                parse_mode='HTML'
            )
        except Exception as send_error:
            logger.error(f"Error sending message in return_to_main_menu: {send_error}")
    finally:
        safe_answer_callback_query(call)

@bot.callback_query_handler(func=lambda call: call.data == 'check_model')
def check_model_status(call: types.CallbackQuery):
    """Проверка статуса модели"""
    try:
        status = get_model_status()
        bot.answer_callback_query(call.id, status, show_alert=True)
    except Exception as e:
        logger.error(f"Ошибка в check_model_status: {e}")
        bot.answer_callback_query(call.id, "⚠️ Ошибка проверки статуса")

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message: types.Message):
    if not is_actual_message(message):
        return
    # Здесь можно обработать любые другие сообщения, если нужно
    pass

# ==================== ЗАПУСК БОТА ====================
if __name__ == '__main__':
    print(""" 
    ██████╗  ██╗      █████╗ ███████╗ █████╗     ██████╗  ██████╗ ██████╗ 
    ██╔════╝ ██║     ██╔══██╗╚══███╔╝██╔══██╗    ██╔══██╗██╔═══██╗██╔══██╗
    ██║  ███╗██║     ███████║  ███╔╝ ███████║    ██║  ██║██║   ██║██████╔╝
    ██║   ██║██║     ██╔══██║ ███╔╝  ██╔══██║    ██║  ██║██║   ██║██╔══██╗
    ╚██████╔╝███████╗██║  ██║███████╗██║  ██║    ██████╔╝╚██████╔╝██║  ██║
     ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═════╝  ╚═════╝ ╚═╝  ╚═╝
    """)
    
    logger.info(f"🟢 Запуск бота @{CONFIG.bot_username}...")
    
    # Создаем необходимые папки
    os.makedirs(CONFIG.data_folder, exist_ok=True)
    os.makedirs(CONFIG.backup_folder, exist_ok=True)
    os.makedirs(CONFIG.reports_folder, exist_ok=True)
    os.makedirs(CONFIG.support_folder, exist_ok=True)
    
    # Загружаем модель только после запуска бота
    logger.info("🔄 Начало загрузки модели...")
    load_model()
    
    # Основной цикл бота
    while True:
        try:
            logger.info("Запуск polling...")
            bot.infinity_polling(timeout=60, long_polling_timeout=60)
        except ApiTelegramException as e:
            logger.error(f"Ошибка Telegram API: {e}")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            traceback.print_exc()
            time.sleep(30)
        finally:
            user_db.close()
            logger.info("Бот остановлен, перезапуск через 5 секунд...")
            time.sleep(5)
