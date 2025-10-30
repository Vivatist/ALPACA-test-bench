#!/usr/bin/env python3
"""
Основной startup скрипт для ALPACA Test Bench.
"""

import sys
import argparse
from pathlib import Path

# Добавляем src в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """Главная функция запуска."""
    parser = argparse.ArgumentParser(
        description="ALPACA Test Bench - Испытательный стенд для тестирования библиотек оцифровки документов"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Streamlit UI
    ui_parser = subparsers.add_parser('ui', help='Запуск веб-интерфейса')
    ui_parser.add_argument('--port', type=int, default=8501, help='Порт для веб-интерфейса')
    ui_parser.add_argument('--host', default='localhost', help='Хост для веб-интерфейса')
    
    # CLI команды
    cli_parser = subparsers.add_parser('cli', help='Использование CLI интерфейса')
    cli_parser.add_argument('cli_args', nargs='*', help='Аргументы для CLI')
    
    # Настройка
    setup_parser = subparsers.add_parser('setup', help='Первоначальная настройка системы')
    setup_parser.add_argument('--force', action='store_true', help='Принудительно пересоздать конфигурацию')
    
    # Проверка зависимостей
    check_parser = subparsers.add_parser('check', help='Проверка зависимостей и окружения')
    check_parser.add_argument('--install', action='store_true', help='Попытаться установить недостающие зависимости')
    
    args = parser.parse_args()
    
    if args.command == 'ui':
        run_streamlit_ui(args.host, args.port)
    elif args.command == 'cli':
        run_cli(args.cli_args)
    elif args.command == 'setup':
        setup_system(args.force)
    elif args.command == 'check':
        check_dependencies(args.install)
    else:
        parser.print_help()


def run_streamlit_ui(host: str = 'localhost', port: int = 8501):
    """Запускает Streamlit интерфейс."""
    import subprocess
    import sys
    
    print(f"🚀 Запуск веб-интерфейса на http://{host}:{port}")
    print("Для остановки нажмите Ctrl+C")
    
    try:
        streamlit_app = project_root / "src" / "ui" / "streamlit_app.py"
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(streamlit_app),
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка запуска Streamlit: {e}")
        print("Убедитесь, что Streamlit установлен: pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 Веб-интерфейс остановлен")


def run_cli(cli_args: list):
    """Запускает CLI интерфейс."""
    import sys
    
    try:
        from src.cli import cli
        
        # Передаем аргументы в CLI
        sys.argv = ['alpaca-cli'] + cli_args
        cli()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта CLI: {e}")
        print("Проверьте установку зависимостей")
    except Exception as e:
        print(f"❌ Ошибка CLI: {e}")


def setup_system(force: bool = False):
    """Выполняет первоначальную настройку системы."""
    print("🛠️  Настройка ALPACA Test Bench...")
    
    # Создаем необходимые директории
    directories = [
        "test_files",
        "outputs", 
        "logs",
        "configs"
    ]
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        if not dir_path.exists() or force:
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Создана директория: {dir_path}")
    
    # Создаем конфигурацию
    try:
        sys.path.insert(0, str(project_root / "src"))
        from configs.system_configurator import create_sample_config
        
        config_file = project_root / "configs" / "system_config.json"
        if not config_file.exists() or force:
            create_sample_config()
            print("✅ Создана конфигурация системы")
        else:
            print("ℹ️  Конфигурация уже существует")
    
    except Exception as e:
        print(f"⚠️  Ошибка создания конфигурации: {e}")
    
    # Создаем .env файл
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if env_example.exists() and (not env_file.exists() or force):
        try:
            env_content = env_example.read_text(encoding='utf-8')
            env_file.write_text(env_content, encoding='utf-8')
            print("✅ Создан файл окружения .env")
        except Exception as e:
            print(f"⚠️  Ошибка создания .env: {e}")
    
    print("\n🎉 Настройка завершена!")
    print("\nДля запуска веб-интерфейса выполните: python main.py ui")
    print("Для справки по CLI: python main.py cli --help")


def check_dependencies(install: bool = False):
    """Проверяет зависимости и окружение."""
    print("🔍 Проверка зависимостей...")
    
    required_packages = [
        'streamlit',
        'fastapi', 
        'pandas',
        'plotly',
        'click',
        'python-docx',
        'PyPDF2',
        'pdfplumber'
    ]
    
    optional_packages = [
        'pymupdf',
        'pdfminer.six', 
        'camelot-py',
        'tesseract',
        'easyocr',
        'markdownify',
        'html2text',
        'beautifulsoup4'
    ]
    
    missing_required = []
    missing_optional = []
    
    # Проверяем обязательные пакеты
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package} (обязательный)")
    
    # Проверяем дополнительные пакеты
    for package in optional_packages:
        try:
            if package == 'pymupdf':
                __import__('fitz')
            elif package == 'tesseract':
                # Проверяем исполняемый файл tesseract
                import shutil
                if shutil.which('tesseract'):
                    print(f"✅ {package}")
                else:
                    raise ImportError()
            else:
                __import__(package.replace('-', '_'))
                print(f"✅ {package} (дополнительный)")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️  {package} (дополнительный)")
    
    # Проверяем Python версию
    if sys.version_info >= (3, 9):
        print(f"✅ Python {sys.version}")
    else:
        print(f"⚠️  Python {sys.version} (рекомендуется 3.9+)")
    
    # Результаты
    print(f"\n📊 Сводка:")
    print(f"✅ Установлено обязательных: {len(required_packages) - len(missing_required)}/{len(required_packages)}")
    print(f"✅ Установлено дополнительных: {len(optional_packages) - len(missing_optional)}/{len(optional_packages)}")
    
    if missing_required:
        print(f"\n❌ Отсутствуют обязательные пакеты:")
        for package in missing_required:
            print(f"  • {package}")
        
        if install:
            print("\n📦 Установка недостающих пакетов...")
            install_packages(missing_required)
    
    if missing_optional:
        print(f"\n⚠️  Отсутствуют дополнительные пакеты:")
        for package in missing_optional:
            print(f"  • {package}")
        
        if install:
            print("\n📦 Установка дополнительных пакетов...")
            install_packages(missing_optional, optional=True)
    
    # Проверяем специальные зависимости
    check_special_dependencies()


def install_packages(packages: list, optional: bool = False):
    """Устанавливает пакеты через pip."""
    import subprocess
    import sys
    
    for package in packages:
        try:
            print(f"Установка {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ Установлен {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка установки {package}: {e}")
            if not optional:
                print(f"Установите вручную: pip install {package}")


def check_special_dependencies():
    """Проверяет специальные зависимости (Tesseract, LibreOffice и т.д.)."""
    print("\n🔍 Проверка специальных зависимостей:")
    
    # Tesseract OCR
    import shutil
    if shutil.which('tesseract'):
        try:
            import subprocess
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            version = result.stdout.split('\n')[0]
            print(f"✅ {version}")
        except:
            print("⚠️  Tesseract установлен, но возможны проблемы с версией")
    else:
        print("❌ Tesseract OCR не найден")
        print("   Установка: https://github.com/UB-Mannheim/tesseract/wiki")
    
    # LibreOffice (для .doc файлов)
    libreoffice_paths = [
        'libreoffice', 'soffice',
        'C:\\Program Files\\LibreOffice\\program\\soffice.exe'
    ]
    
    libreoffice_found = False
    for path in libreoffice_paths:
        if shutil.which(path) or Path(path).exists():
            print("✅ LibreOffice найден")
            libreoffice_found = True
            break
    
    if not libreoffice_found:
        print("⚠️  LibreOffice не найден (для обработки .doc файлов)")
        print("   Установка: https://www.libreoffice.org/download/")
    
    # Pandoc (для продвинутой конвертации)
    if shutil.which('pandoc'):
        print("✅ Pandoc найден")
    else:
        print("⚠️  Pandoc не найден (дополнительные возможности конвертации)")
        print("   Установка: https://pandoc.org/installing.html")


if __name__ == "__main__":
    main()