import logging
import sys

def setup_logger():
    """
    Konfiguruje logger:
    - Konsola: Tylko najważniejsze informacje (INFO i wyżej).
    """
    # Wymuszenie UTF-8 na Windows (konsola PowerShell domyślnie używa cp1250)
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # Pobierz root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Wyczyść istniejące handlery (aby uniknąć duplikacji przy reloadzie)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Handler konsoli 
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(message)s') # Czysty format dla konsoli
    c_handler.setFormatter(c_format)

    # Dodanie handlera
    logger.addHandler(c_handler)
    
    return logger
