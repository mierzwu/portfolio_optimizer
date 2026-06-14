"""
gui.py – Graficzny interfejs użytkownika dla Optymalizatora Portfela.

Ekran 1 – Wprowadzanie parametrów (portfel, parametry optymalizacji, ograniczenia).
Ekran 2 – Wyniki (metryki portfela, optymalne wagi, transakcje, symulacje, stress-testy, log).
"""
import sys
import json
import traceback
import logging
import time
from datetime import date
from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QTextEdit, QMessageBox,
    QTabWidget, QProgressBar, QAbstractItemView,
    QDialog, QDialogButtonBox, QSpinBox, QDateEdit, QFrame,
    QToolTip,
)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer, QDate
from PySide6.QtGui import QFont, QCursor

logger = logging.getLogger(__name__)


# ─── Logging bridge ──────────────────────────────────────────────────────────

class _LogEmitter(QObject):
    message = Signal(str)


class _QTextEditHandler(logging.Handler):
    """Forwards log records to a QTextEdit via Qt signals (thread-safe)."""

    def __init__(self, emitter: _LogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord):
        self.emitter.message.emit(self.format(record))


# ─── Background worker ────────────────────────────────────────────────────────

class OptimizationWorker(QObject):
    progress = Signal(str)
    result_ready = Signal(dict)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, validated_data):
        super().__init__()
        self.validated_data = validated_data

    def run(self):
        try:
            from data_fetcher import fetch_market_data
            from data_processor import preprocess_data
            from parameter_estimator import estimate_params
            from optimizer import optimize_portfolio
            from backtester import backtest_and_simulate

            self.progress.emit("Pobieranie danych rynkowych…")
            logger.info("[Pipeline] Krok 1/6: Pobieranie danych rynkowych")
            market_data = fetch_market_data(self.validated_data)
            logger.info("[Pipeline] Dane rynkowe pobrane")

            self.progress.emit("Przetwarzanie danych…")
            logger.info("[Pipeline] Krok 2/6: Przetwarzanie danych")
            processed_data = preprocess_data(market_data, self.validated_data.data_policy)
            logger.info("[Pipeline] Dane przetworzone")

            self.progress.emit("Estymacja parametrów modelu…")
            logger.info("[Pipeline] Krok 3/6: Estymacja parametrów")
            model_params = estimate_params(
                processed_data,
                self.validated_data.estimation_window,
                self.validated_data.investment_horizon_years,
            )
            logger.info("[Pipeline] Parametry wyestymowane")

            self.progress.emit("Analiza bieżącego portfela…")
            logger.info("[Pipeline] Krok 4/6: Analiza bieżącego portfela")
            current_portfolio = self._compute_current_portfolio(model_params, processed_data)
            logger.info("[Pipeline] Analiza portfela zakończona")

            self.progress.emit("Optymalizacja portfela…")
            logger.info("[Pipeline] Krok 5/6: Optymalizacja portfela")
            optimization = optimize_portfolio(self.validated_data, model_params, processed_data)
            logger.info("[Pipeline] Optymalizacja zakończona")

            sim_results = None
            if optimization:
                self.progress.emit("Symulacje Monte Carlo i stress-testy…")
                logger.info("[Pipeline] Krok 6/6: Symulacje Monte Carlo i stress-testy")
                sim_results = backtest_and_simulate(
                    self.validated_data, processed_data, optimization, model_params
                )
                logger.info("[Pipeline] Symulacje zakończone")

            # Zapis do bazy danych (zawsze, gdy optymalizacja się powiodła)
            if optimization:
                self.progress.emit("Zapisywanie wyników do bazy danych…")
                try:
                    from database import initialize_db, save_full_result
                    initialize_db()
                    bond_tickers = {
                        item.ticker for item in self.validated_data.portfolio
                        if item.instrument_type.value == "bond"
                    }
                    cvar_alpha = self.validated_data.parametry_opt.cvar_alpha or 0.99
                    goal = self.validated_data.parametry_opt.goal_type.value
                    horyzont = self.validated_data.investment_horizon_years
                    prowizja = self.validated_data.ustawienia_ograniczen.transaction_cost_pct
                    from datetime import datetime as _dt
                    nazwa = f"{_dt.now().strftime('%Y-%m-%d %H:%M')} | {goal.upper()} | {horyzont}L"
                    save_full_result(
                        optimization_result=optimization,
                        bond_tickers=bond_tickers,
                        nazwa_strategii=nazwa,
                        cel_optymalizacji=goal,
                        horyzont_inwestycyjny_lat=horyzont,
                        prowizja_maklerska=prowizja,
                        cvar_alpha=cvar_alpha,
                    )
                except Exception as db_exc:
                    logger.warning(f"[DB] Błąd zapisu wyników: {db_exc}")

            self.progress.emit("Gotowe.")
            self.result_ready.emit({
                "current_portfolio": current_portfolio,
                "optimization": optimization,
                "simulation": sim_results,
                "tickers": model_params["mu"].index.tolist(),
            })

        except Exception as exc:
            self.error_occurred.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )
        finally:
            self.finished.emit()

    def _compute_current_portfolio(self, model_params, processed_data) -> dict:
        last_date = model_params["last_date"]
        current_prices = processed_data["prices"].loc[last_date]
        tickers = model_params["mu"].index.tolist()
        holdings = {item.ticker: item.quantity for item in self.validated_data.portfolio}

        w_vals = np.array([
            holdings.get(t, 0.0) * current_prices.get(t, 0.0) for t in tickers
        ])
        cash = max(self.validated_data.additional_cash or 0.0, 0.0)
        invested_value = float(w_vals.sum())
        total_value = invested_value + cash  # łączny majątek (do wyświetlenia)

        # Metryki bieżącego portfela liczone na ZAINWESTOWANEJ części (bez gotówki).
        # Gotówka nie rozcieńcza metryki — chcemy wiedzieć, jak działa bieżąca inwestycja,
        # a celem optymalizacji jest utrzymanie tego zwrotu po wdrożeniu gotówki.
        w_invested = w_vals / invested_value if invested_value > 0 else np.zeros(len(tickers))
        sigma = model_params.get("sigma_shrink", model_params["sigma"]).values
        rf = model_params.get("current_cpi", 0.0)

        # additional_cash=0: zwrot bieżącego portfela bez rozcieńczenia gotówką
        ret = _compute_portfolio_expected_return(
            model_params, processed_data, self.validated_data.portfolio,
        ) or 0.0
        vol = float(np.sqrt(w_invested @ sigma @ w_invested)) if invested_value > 0 else 0.0
        sharpe = (ret - rf) / vol if vol > 0 else 0.0

        return {
            "total_value": total_value,
            "quantities": {item.ticker: item.quantity for item in self.validated_data.portfolio},
            "weights": dict(zip(tickers, w_invested.tolist())),
            "metrics": {
                "expected_return": ret,
                "volatility": vol,
                "sharpe_ratio": sharpe,
            },
        }


# ─── Shared helper: compute expected return of a portfolio ──────────────────

def _compute_portfolio_expected_return(
    model_params: dict, processed_data: dict, portfolio, additional_cash: float = 0.0
) -> Optional[float]:
    """Returns annualised expected return of *portfolio* using the same method
    as the optimiser's current-portfolio analysis.  Returns None if the
    portfolio has zero value in the model (e.g. all tickers missing prices).
    additional_cash is included in total_value as undeployed capital earning 0%,
    diluting the current return proportionally."""
    last_date = model_params["last_date"]
    current_prices = processed_data["prices"].loc[last_date]
    tickers = model_params["mu"].index.tolist()
    holdings = {item.ticker: item.quantity for item in portfolio}
    w_vals = np.array([
        holdings.get(t, 0.0) * current_prices.get(t, 0.0) for t in tickers
    ])
    total_value = float(w_vals.sum()) + max(additional_cash, 0.0)
    if total_value <= 0:
        return None
    w = w_vals / total_value
    return float(w @ model_params["mu"].values)


# ─── Background thread: current portfolio return ─────────────────────────────

class CurrentReturnThread(QThread):
    """QThread subclass – fetches market data and emits the current portfolio
    expected return.  Using a subclass (instead of worker+moveToThread) avoids
    Qt ownership/lifetime issues that cause 'QThread destroyed while running'."""
    result_ready = Signal(float)

    def __init__(self, validated_data, parent=None):
        super().__init__(parent)
        self.validated_data = validated_data

    def run(self):
        try:
            from data_fetcher import fetch_market_data
            from data_processor import preprocess_data
            from parameter_estimator import estimate_params

            market_data = fetch_market_data(self.validated_data)
            processed_data = preprocess_data(market_data, self.validated_data.data_policy)
            model_params = estimate_params(
                processed_data,
                self.validated_data.estimation_window,
                self.validated_data.investment_horizon_years,
            )

            ret = _compute_portfolio_expected_return(
                model_params, processed_data, self.validated_data.portfolio,
            )
            if ret is not None:
                self.result_ready.emit(ret)
        except Exception as exc:
            logger.warning(f"[CurrentReturnThread] Błąd obliczania stopy zwrotu: {exc}", exc_info=True)
        # QThread.finished is emitted automatically when run() returns


# ─── Screen 1: Input ──────────────────────────────────────────────────────────

# ─── Help icon widget ────────────────────────────────────────────────────────

class _HelpIcon(QLabel):
    """Circular '?' label that shows a tooltip via QToolTip.showText on hover."""

    def __init__(self, tooltip_text: str, parent=None):
        super().__init__("?", parent)
        self._tooltip_text = tooltip_text
        self.setFixedSize(16, 16)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_Hover, True)
        self.setStyleSheet(
            "QLabel { background: #b0b8c8; color: #1a2540; border-radius: 8px;"
            " font-size: 10px; font-weight: bold; }"
            "QLabel:hover { background: #0055aa; color: white; }"
        )

    def enterEvent(self, event):
        QToolTip.showText(
            self.mapToGlobal(self.rect().center()),
            self._tooltip_text,
            self,
        )
        super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


def _make_label_with_help(label_text: str, tooltip_text: str) -> QWidget:
    """Returns a widget with a text label and a '?' help icon showing a tooltip on hover."""
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    lbl = QLabel(label_text)
    row.addWidget(lbl)
    row.addWidget(_HelpIcon(tooltip_text))
    row.addStretch()
    return w


# ─── Dialog: edycja / dodawanie emisji obligacji ──────────────────────────────
class BondEmissionDialog(QDialog):
    def __init__(self, parent=None, data: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edytuj emisję" if data else "Nowa emisja obligacji")
        self.setMinimumWidth(440)
        layout = QFormLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.combo_typ = QComboBox()
        self.combo_typ.addItems(["EDO", "COI"])
        if data:
            idx = self.combo_typ.findText(data.get("typ_obligacji", "EDO"))
            if idx >= 0:
                self.combo_typ.setCurrentIndex(idx)
        layout.addRow("Typ:", self.combo_typ)

        self.edit_symbol = QLineEdit(data["symbol_emisji"] if data else "")
        self.edit_symbol.setPlaceholderText("np. EDO0536")
        if data:
            self.edit_symbol.setReadOnly(True)
        layout.addRow("Symbol:", self.edit_symbol)

        self.edit_date_start = QLineEdit(data["data_poczatkowa"] if data else "")
        self.edit_date_start.setPlaceholderText("RRRR-MM-DD")
        layout.addRow("Początek emisji:", self.edit_date_start)

        self.edit_date_end = QLineEdit(data["data_zakonczenia"] if data else "")
        self.edit_date_end.setPlaceholderText("RRRR-MM-DD")
        layout.addRow("Koniec emisji:", self.edit_date_end)

        self.spin_years = QSpinBox()
        self.spin_years.setRange(1, 40)
        self.spin_years.setValue(int(data["dlugosc_lat"]) if data and "dlugosc_lat" in data else 10)
        layout.addRow("Długość (lata):", self.spin_years)

        self.edit_rate = QLineEdit(str(data["oprocentowanie_rok_1"]) if data else "")
        self.edit_rate.setPlaceholderText("np. 0.0535")
        layout.addRow("Oprocentowanie rok 1:", self.edit_rate)

        self.edit_margin = QLineEdit(str(data["marza_odsetkowa"]) if data else "")
        self.edit_margin.setPlaceholderText("np. 0.02")
        layout.addRow("Marża odsetkowa:", self.edit_margin)

        self.edit_penalty = QLineEdit(str(data["kara_wykup"]) if data else "")
        self.edit_penalty.setPlaceholderText("np. 3.0")
        layout.addRow("Kara za wykup (PLN):", self.edit_penalty)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _validate_and_accept(self):
        errors = []
        if not self.edit_symbol.text().strip():
            errors.append("Symbol nie może być pusty.")
        if not self.edit_date_start.text().strip() or not self.edit_date_end.text().strip():
            errors.append("Daty muszą być podane w formacie RRRR-MM-DD.")
        for field, label in [
            (self.edit_rate, "Oprocentowanie"),
            (self.edit_margin, "Marża"),
            (self.edit_penalty, "Kara"),
        ]:
            try:
                float(field.text())
            except ValueError:
                errors.append(f"{label} musi być liczbą.")
        if errors:
            QMessageBox.warning(self, "Błąd", "\n".join(errors))
            return
        self.accept()

    def get_data(self) -> dict:
        return {
            "typ": self.combo_typ.currentText(),
            "symbol": self.edit_symbol.text().strip().upper(),
            "data_poczatkowa": self.edit_date_start.text().strip(),
            "data_zakonczenia": self.edit_date_end.text().strip(),
            "dlugosc_lat": self.spin_years.value(),
            "oprocentowanie_rok_1": float(self.edit_rate.text()),
            "marza": float(self.edit_margin.text()),
            "kara": float(self.edit_penalty.text()),
        }


# ─── Screen 3: Baza danych ────────────────────────────────────────────────────
class BondsScreen(QWidget):
    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bonds_data = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        btn_back = QPushButton("← Powrót")
        btn_back.setFixedWidth(120)
        btn_back.clicked.connect(self.back_requested)
        lbl = QLabel("Zbiór obligacji")
        lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
        header.addWidget(btn_back)
        header.addSpacing(12)
        header.addWidget(lbl)
        header.addStretch()
        root.addLayout(header)

        toolbar = QHBoxLayout()
        btn_add = QPushButton("+ Dodaj")
        btn_edit = QPushButton("✎ Edytuj")
        btn_delete = QPushButton("✕ Usuń")
        btn_refresh = QPushButton("⟳ Odśwież")
        for b in (btn_add, btn_edit, btn_delete, btn_refresh):
            b.setFixedHeight(28)
            toolbar.addWidget(b)
        toolbar.addStretch()
        root.addLayout(toolbar)

        self.tbl_bonds = QTableWidget()
        self.tbl_bonds.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_bonds.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_bonds.setAlternatingRowColors(True)
        root.addWidget(self.tbl_bonds)

        btn_add.clicked.connect(self._add_bond)
        btn_edit.clicked.connect(self._edit_bond)
        btn_delete.clicked.connect(self._delete_bond)
        btn_refresh.clicked.connect(self._load_bonds)

    def _load_bonds(self):
        from database import get_bond_emissions
        rows = get_bond_emissions()
        headers = ["Typ", "Symbol", "Początek emisji", "Koniec emisji", "Długość (lata)", "Oprocent. r.1", "Marża", "Kara (PLN)"]
        self.tbl_bonds.setColumnCount(len(headers))
        self.tbl_bonds.setHorizontalHeaderLabels(headers)
        self.tbl_bonds.setRowCount(len(rows))
        for r, b in enumerate(rows):
            self.tbl_bonds.setItem(r, 0, QTableWidgetItem(b["typ_obligacji"]))
            self.tbl_bonds.setItem(r, 1, QTableWidgetItem(b["symbol_emisji"]))
            self.tbl_bonds.setItem(r, 2, QTableWidgetItem(b["data_poczatkowa"]))
            self.tbl_bonds.setItem(r, 3, QTableWidgetItem(b["data_zakonczenia"]))
            self.tbl_bonds.setItem(r, 4, QTableWidgetItem(str(b.get("dlugosc_lat", ""))))
            self.tbl_bonds.setItem(r, 5, QTableWidgetItem(f"{b['oprocentowanie_rok_1']:.4f}"))
            self.tbl_bonds.setItem(r, 6, QTableWidgetItem(f"{b['marza_odsetkowa']:.4f}"))
            self.tbl_bonds.setItem(r, 7, QTableWidgetItem(f"{b['kara_wykup']:.2f}"))
        self.tbl_bonds.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._bonds_data = rows

    def _add_bond(self):
        from database import upsert_bond_emission
        dlg = BondEmissionDialog(self)
        if dlg.exec() == QDialog.Accepted:
            d = dlg.get_data()
            upsert_bond_emission(**d)
            self._load_bonds()

    def _edit_bond(self):
        row = self.tbl_bonds.currentRow()
        if row < 0:
            QMessageBox.information(self, "Info", "Wybierz emisję do edycji.")
            return
        from database import upsert_bond_emission
        bond = self._bonds_data[row]
        dlg = BondEmissionDialog(self, data=bond)
        if dlg.exec() == QDialog.Accepted:
            d = dlg.get_data()
            upsert_bond_emission(**d)
            self._load_bonds()

    def _delete_bond(self):
        row = self.tbl_bonds.currentRow()
        if row < 0:
            QMessageBox.information(self, "Info", "Wybierz emisję do usunięcia.")
            return
        bond = self._bonds_data[row]
        reply = QMessageBox.question(
            self, "Potwierdź",
            f"Usunąć emisję '{bond['symbol_emisji']}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            from database import delete_bond_emission
            delete_bond_emission(bond["symbol_emisji"])
            self._load_bonds()


class HistoryScreen(QWidget):
    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history_data = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        btn_back = QPushButton("← Powrót")
        btn_back.setFixedWidth(120)
        btn_back.clicked.connect(self.back_requested)
        lbl = QLabel("Historia analiz")
        lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
        header.addWidget(btn_back)
        header.addSpacing(12)
        header.addWidget(lbl)
        header.addStretch()
        root.addLayout(header)

        toolbar = QHBoxLayout()
        btn_delete = QPushButton("✕ Usuń analizę")
        btn_refresh = QPushButton("⟳ Odśwież")
        for b in (btn_delete, btn_refresh):
            b.setFixedHeight(28)
            toolbar.addWidget(b)
        toolbar.addStretch()
        root.addLayout(toolbar)

        self.tbl_history = QTableWidget()
        self.tbl_history.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_history.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_history.setAlternatingRowColors(True)
        root.addWidget(self.tbl_history, stretch=2)

        grp = QGroupBox("Wagi portfela wybranej analizy")
        self._hist_weights_layout = QVBoxLayout(grp)
        root.addWidget(grp, stretch=1)

        self.tbl_history.itemSelectionChanged.connect(self._show_weights)
        btn_delete.clicked.connect(self._delete_analysis)
        btn_refresh.clicked.connect(self._load_history)

    def _load_history(self):
        from database import initialize_db, get_analysis_history
        initialize_db()
        rows = get_analysis_history(limit=200)
        headers = ["Nazwa analizy", "Data", "Cel", "Horyzont (l.)", "Prowizja", "α CVaR", "Oczek. zwrot", "CVaR", "Koszt rebalan."]
        self.tbl_history.setColumnCount(len(headers))
        self.tbl_history.setHorizontalHeaderLabels(headers)
        self.tbl_history.setRowCount(len(rows))
        for r, h in enumerate(rows):
            self.tbl_history.setItem(r, 0, QTableWidgetItem(h["nazwa_strategii"]))
            self.tbl_history.setItem(r, 1, QTableWidgetItem(h["data_analizy"]))
            self.tbl_history.setItem(r, 2, QTableWidgetItem(h["cel_optymalizacji"]))
            self.tbl_history.setItem(r, 3, QTableWidgetItem(str(h["horyzont_inwestycyjny_lat"])))
            _prow = h['prowizja_maklerska']
            self.tbl_history.setItem(r, 4, QTableWidgetItem(f"{_prow:.4f}" if _prow is not None else "—"))
            self.tbl_history.setItem(r, 5, QTableWidgetItem(f"{h['cvar_alpha']:.2f}"))
            self.tbl_history.setItem(r, 6, QTableWidgetItem(f"{h['oczekiwana_stopa_zwrotu']:.2%}"))
            self.tbl_history.setItem(r, 7, QTableWidgetItem(f"{h['wartosc_ryzyka_cvar']:.2%}"))
            _koszt = h['koszt_rebalancingu_netto']
            self.tbl_history.setItem(r, 8, QTableWidgetItem(f"{_koszt:.2f} PLN" if _koszt is not None else "—"))
        self.tbl_history.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._history_data = rows

    def _show_weights(self):
        row = self.tbl_history.currentRow()
        if row < 0 or not self._history_data:
            return
        analiza_id = self._history_data[row]["id"]
        from database import get_portfolio_weights
        weights = get_portfolio_weights(analiza_id)
        while self._hist_weights_layout.count():
            child = self._hist_weights_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        tbl = QTableWidget(len(weights), 3)
        tbl.setHorizontalHeaderLabels(["Ticker", "Klasa", "Waga docelowa"])
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for r, w in enumerate(weights):
            tbl.setItem(r, 0, QTableWidgetItem(w["ticker_aktywa"]))
            tbl.setItem(r, 1, QTableWidgetItem(w["klasa_aktywa"]))
            tbl.setItem(r, 2, QTableWidgetItem(f"{w['waga_docelowa']:.2%}"))
        self._hist_weights_layout.addWidget(tbl)

    def _delete_analysis(self):
        row = self.tbl_history.currentRow()
        if row < 0:
            QMessageBox.information(self, "Info", "Wybierz analizę do usunięcia.")
            return
        analiza = self._history_data[row]
        reply = QMessageBox.question(
            self, "Potwierdź",
            f"Usunąć analizę ID={analiza['id']} ({analiza['data_analizy']})?\n"
            "Powiązane wagi portfela zostaną usunięte automatycznie (CASCADE).",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            with __import__("database").get_connection() as conn:
                conn.execute("DELETE FROM analizy_historia WHERE id = ?", (analiza["id"],))
            self._load_history()
            while self._hist_weights_layout.count():
                child = self._hist_weights_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()


# ─── Dialog: wybór emisji obligacji ───────────────────────────────────────────
# ─── Wątek pobierania ceny akcji ─────────────────────────────────────────────

class _PriceFetchThread(QThread):
    price_ready = Signal(float)
    error = Signal(str)

    def __init__(self, ticker: str, target_date=None, parent=None):
        super().__init__(parent)
        self._ticker = ticker
        self._target_date = target_date  # datetime.date or None

    def run(self):
        try:
            from datetime import date as _date, timedelta
            import yfinance as yf
            import pandas as pd
            yf_ticker = self._ticker if self._ticker.endswith('.WA') else f"{self._ticker}.WA"
            if self._target_date is None:
                end = _date.today()
                start = end - timedelta(days=14)
            else:
                start = self._target_date
                end = self._target_date + timedelta(days=10)
            data = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data.empty:
                self.error.emit(f"Brak danych dla \u2018{self._ticker}\u2019")
                return
            prices = data['Close'] if 'Close' in data.columns else data
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
            prices = prices.dropna()
            if prices.empty:
                self.error.emit(f"Brak danych dla \u2018{self._ticker}\u2019")
                return
            price = float(prices.iloc[0] if self._target_date is not None else prices.iloc[-1])
            self.price_ready.emit(price)
        except Exception as exc:
            self.error.emit(str(exc))


# ─── Dialog dodawania akcji ───────────────────────────────────────────────────

class AddStockDialog(QDialog):
    """Prowadzi użytkownika przez wprowadzenie tickera, daty nabycia i ilości.
    Po podaniu danych automatycznie pobiera cenę z Yahoo Finance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dodaj akcję")
        self.setMinimumWidth(440)
        self._price: Optional[float] = None
        self._fetch_thread: Optional[_PriceFetchThread] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(10)

        # Ticker + przycisk pobierania
        ticker_row = QHBoxLayout()
        self.edit_ticker = QLineEdit()
        self.edit_ticker.setPlaceholderText("np. CDR, PKO, ALLEGRO")
        ticker_row.addWidget(self.edit_ticker)
        self.btn_fetch = QPushButton("Pobierz cenę")
        self.btn_fetch.clicked.connect(self._fetch_price)
        ticker_row.addWidget(self.btn_fetch)
        form.addRow(
            _make_label_with_help(
                "Ticker:",
                "Symbol giełdowy akcji notowanej na GPW.\n"
                "Wpisz skrót bez przyrostka .WA, np. CDR, PKN, PKO.\n"
                "Po wpisaniu kliknij 'Pobierz cenę', aby automatycznie\n"
                "pobrać ostatnią cenę zamknięcia z Yahoo Finance."
            ),
            ticker_row,
        )

        # Opcjonalna data nabycia
        date_row = QHBoxLayout()
        self.chk_date = QCheckBox("Określ datę nabycia")
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setEnabled(False)
        self.chk_date.toggled.connect(self.date_edit.setEnabled)
        date_row.addWidget(self.chk_date)
        date_row.addWidget(self.date_edit)
        date_row.addStretch()
        form.addRow(
            _make_label_with_help(
                "Data nabycia:",
                "Opcjonalna data nabycia akcji (format RRRR-MM-DD).\n"
                "Jeśli zostanie podana, cena zostanie pobrana z tego dnia\n"
                "(pierwsza dostępna sesja od podanej daty).\n"
                "Bez zaznaczenia pobierana jest bieżąca cena rynkowa."
            ),
            date_row,
        )

        layout.addLayout(form)

        # Status pobierania ceny
        self.lbl_price = QLabel("Kliknij \u201ePobierz cenę\u201d, aby uzupełnić cenę nabycia.")
        self.lbl_price.setStyleSheet("color: #555; font-style: italic;")
        self.lbl_price.setWordWrap(True)
        layout.addWidget(self.lbl_price)

        # Ilość
        qty_form = QFormLayout()
        qty_form.setSpacing(10)
        self.spin_qty = QSpinBox()
        self.spin_qty.setRange(1, 9_999_999)
        self.spin_qty.setValue(1)
        qty_form.addRow(
            _make_label_with_help(
                "Ilość (szt.):",
                "Liczba posiadanych sztuk (akcji) danego instrumentu.\n"
                "Wartość musi być liczbą całkowitą większą od zera."
            ),
            self.spin_qty,
        )
        layout.addLayout(qty_form)

        self._btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._btn_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self._btn_box.accepted.connect(self.accept)
        self._btn_box.rejected.connect(self.reject)
        layout.addWidget(self._btn_box)

    def _fetch_price(self):
        ticker = self.edit_ticker.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Błąd", "Podaj ticker przed pobraniem ceny.")
            return
        target_date = None
        if self.chk_date.isChecked():
            qd = self.date_edit.date()
            from datetime import date as _date
            target_date = _date(qd.year(), qd.month(), qd.day())
        self.btn_fetch.setEnabled(False)
        self.lbl_price.setText("Pobieranie…")
        self.lbl_price.setStyleSheet("color: #555; font-style: italic;")
        self._btn_box.button(QDialogButtonBox.Ok).setEnabled(False)
        if self._fetch_thread and self._fetch_thread.isRunning():
            self._fetch_thread.quit()
            self._fetch_thread.wait(2000)
        self._fetch_thread = _PriceFetchThread(ticker, target_date, self)
        self._fetch_thread.price_ready.connect(self._on_price)
        self._fetch_thread.error.connect(self._on_error)
        self._fetch_thread.finished.connect(lambda: self.btn_fetch.setEnabled(True))
        self._fetch_thread.start()

    def _on_price(self, price: float):
        self._price = round(price, 2)
        self.edit_ticker.setText(self.edit_ticker.text().strip().upper())
        self.lbl_price.setText(f"Cena nabycia: {self._price:,.2f} PLN")
        self.lbl_price.setStyleSheet("color: #1a7a1a; font-weight: bold;")
        self._btn_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def _on_error(self, msg: str):
        self._price = None
        self.lbl_price.setText(f"Błąd pobierania: {msg}")
        self.lbl_price.setStyleSheet("color: #cc0000;")
        self._btn_box.button(QDialogButtonBox.Ok).setEnabled(False)

    def get_data(self) -> dict | None:
        ticker = self.edit_ticker.text().strip().upper()
        if not ticker or self._price is None:
            return None
        from datetime import date as _date
        if self.chk_date.isChecked():
            qd = self.date_edit.date()
            date_str = qd.toString("yyyy-MM-dd")
        else:
            date_str = _date.today().isoformat()
        return {
            "ticker": ticker,
            "quantity": self.spin_qty.value(),
            "date_acquired": date_str,
            "price_acquired": self._price,
        }


class SelectBondDialog(QDialog):
    """
    Pozwala użytkownikowi wybrać emisję obligacji (EDO/COI) z bazy danych.
    Wyświetla listę rozwijaną z dostępnymi emisjami i pozwala wprowadzić ilość.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wybierz emisję obligacji")
        self.setMinimumWidth(480)
        self._emissions: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(10)

        self.combo_emission = QComboBox()
        self.combo_emission.setMinimumWidth(360)
        form.addRow(
            _make_label_with_help(
                "Emisja:",
                "Wybierz serię obligacji skarbowych z bazy danych.\n"
                "Lista zawiera emisje EDO (10-letnie) i COI (4-letnie)\n"
                "wprowadzone w ekranie 'Zbiór obligacji'.\n"
                "Symbol zawiera typ i datę wykupu, np. EDO0536 = EDO maj 2036."
            ),
            self.combo_emission,
        )

        self.spin_qty = QSpinBox()
        self.spin_qty.setRange(1, 999999)
        self.spin_qty.setValue(100)
        form.addRow(
            _make_label_with_help(
                "Ilość (szt.):",
                "Liczba posiadanych sztuk (obligacji) danej emisji.\n"
                "Każda obligacja ma wartość nominalną 100 PLN."
            ),
            self.spin_qty,
        )

        self.edit_date_acq = QLineEdit()
        self.edit_date_acq.setPlaceholderText("RRRR-MM-DD (opcjonalnie)")
        form.addRow(
            _make_label_with_help(
                "Data nabycia:",
                "Data nabycia obligacji w formacie RRRR-MM-DD (opcjonalna).\n"
                "Wymagana dla emisji, których termin wykupu już minął.\n"
                "Musi zawierać się w przedziale\n"
                "[data emisji, data wykupu] danej serii."
            ),
            self.edit_date_acq,
        )

        layout.addLayout(form)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(self.lbl_info)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_emissions()
        self.combo_emission.currentIndexChanged.connect(self._update_info)

    def _load_emissions(self):
        try:
            from database import get_bond_emissions, initialize_db
            initialize_db()
            self._emissions = get_bond_emissions()
        except Exception:
            self._emissions = []

        self.combo_emission.clear()
        if not self._emissions:
            self.combo_emission.addItem("— brak emisji w bazie danych —")
        else:
            for e in self._emissions:
                label = (
                    f"{e['symbol_emisji']}  [{e['typ_obligacji']}]"
                    f"  {e['data_poczatkowa']} → {e['data_zakonczenia']}"
                )
                self.combo_emission.addItem(label)
        self._update_info()

    def _update_info(self):
        e = self._selected_emission()
        if e is None:
            self.lbl_info.setText("")
            return
        self.lbl_info.setText(
            f"Stopa 1. roku: {e['oprocentowanie_rok_1']:.2%}  |  "
            f"Marża: {e['marza_odsetkowa']:.2%}  |  "
            f"Kara wykupu: {e['kara_wykup']:.2f} PLN"
        )

    def _selected_emission(self):
        idx = self.combo_emission.currentIndex()
        if 0 <= idx < len(self._emissions):
            return self._emissions[idx]
        return None

    def _validate_and_accept(self):
        from datetime import date as _date
        e = self._selected_emission()
        if e is None:
            self.accept()
            return

        date_acq_text = self.edit_date_acq.text().strip()
        today = _date.today()

        try:
            end_date = _date.fromisoformat(str(e["data_zakonczenia"]).strip()[:10])
        except (ValueError, TypeError):
            end_date = None

        try:
            start_date = _date.fromisoformat(str(e["data_poczatkowa"]).strip()[:10])
        except (ValueError, TypeError):
            start_date = None

        if end_date and end_date <= today:
            if not date_acq_text:
                QMessageBox.warning(
                    self, "Wymagana data nabycia",
                    f"Emisja {e['symbol_emisji']} zakończyła się {e['data_zakonczenia']} "
                    "(w przeszłości). Podaj datę nabycia obligacji."
                )
                self.edit_date_acq.setFocus()
                return

        if date_acq_text:
            try:
                acq_date = _date.fromisoformat(date_acq_text.strip()[:10])
            except ValueError:
                QMessageBox.warning(
                    self, "Nieprawidłowa data",
                    "Data nabycia musi być w formacie RRRR-MM-DD."
                )
                self.edit_date_acq.setFocus()
                return
        else:
            acq_date = today

        if start_date and acq_date < start_date:
            QMessageBox.warning(
                self, "Nieprawidłowa data nabycia",
                f"Data nabycia ({acq_date}) nie może być wcześniejsza "
                f"niż początek emisji ({e['data_poczatkowa']})."
            )
            if date_acq_text:
                self.edit_date_acq.setFocus()
            return

        if end_date and acq_date > end_date:
            QMessageBox.warning(
                self, "Nieprawidłowa data nabycia",
                f"Data nabycia ({acq_date}) nie może być późniejsza "
                f"niż koniec emisji ({e['data_zakonczenia']})."
            )
            if date_acq_text:
                self.edit_date_acq.setFocus()
            return

        self.accept()

    def get_data(self) -> dict | None:
        """Zwraca słownik z danymi wybranej emisji lub None jeśli brak wyboru."""
        from datetime import date as _date
        e = self._selected_emission()
        if e is None:
            return None
        date_acq_text = self.edit_date_acq.text().strip()
        date_acquired = date_acq_text if date_acq_text else _date.today().isoformat()
        return {
            "ticker": e["symbol_emisji"],
            "symbol_emisji": e["symbol_emisji"],
            "bond_type": e["typ_obligacji"],
            "quantity": self.spin_qty.value(),
            "issue_date": e["data_poczatkowa"],
            "date_acquired": date_acquired,
            "margin": e["marza_odsetkowa"],
            "first_year_rate": e["oprocentowanie_rok_1"],
        }


class InputScreen(QWidget):
    run_requested = Signal(dict)
    compute_current_return_requested = Signal(dict)
    go_bonds_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_defaults()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        title = QLabel("Optymalizator Portfela")
        title.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        root.addWidget(self._build_step_indicator())

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_portfolio_tab())    # 0 – Portfel
        self.pages.addWidget(self._build_params_tab())       # 1 – Parametry
        self.pages.addWidget(self._build_constraints_tab())  # 2 – Ograniczenia
        root.addWidget(self.pages, stretch=1)

        nav = QHBoxLayout()
        self.btn_back = QPushButton("← Wstecz")
        self.btn_back.setMinimumHeight(40)
        self.btn_back.clicked.connect(self._on_prev_step)
        self.btn_next = QPushButton("Dalej →")
        self.btn_next.setMinimumHeight(40)
        f_next = QFont()
        f_next.setBold(True)
        self.btn_next.setFont(f_next)
        self.btn_next.clicked.connect(self._on_next_step)
        self.btn_run = QPushButton("▶  Optymalizuj portfel")
        self.btn_run.setMinimumHeight(50)
        f2 = QFont()
        f2.setPointSize(13)
        f2.setBold(True)
        self.btn_run.setFont(f2)
        self.btn_run.clicked.connect(self._on_run)
        nav.addWidget(self.btn_back)
        nav.addStretch()
        nav.addWidget(self.btn_next)
        nav.addWidget(self.btn_run)
        root.addLayout(nav)

        self._current_step = 0
        self._update_step_ui()

    def _build_step_indicator(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)
        self._step_labels: list[QLabel] = []
        for i, text in enumerate(["1. Portfel", "2. Parametry", "3. Ograniczenia"]):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumWidth(110)
            layout.addWidget(lbl)
            self._step_labels.append(lbl)
            if i < 2:
                sep = QLabel("›")
                sep.setAlignment(Qt.AlignCenter)
                layout.addWidget(sep)
        layout.addStretch()
        return w

    def _update_step_ui(self):
        for i, lbl in enumerate(self._step_labels):
            if i == self._current_step:
                lbl.setStyleSheet("font-weight: bold; color: #0055aa; text-decoration: underline;")
            elif i < self._current_step:
                lbl.setStyleSheet("color: #999999;")
            else:
                lbl.setStyleSheet("color: #cccccc;")
        self.btn_back.setVisible(self._current_step > 0)
        self.btn_next.setVisible(self._current_step < 2)
        self.btn_run.setVisible(self._current_step == 2)
        self.pages.setCurrentIndex(self._current_step)

    def _on_next_step(self):
        if self._current_step == 0:
            if self.stock_table.rowCount() == 0 and self.bond_table.rowCount() == 0:
                QMessageBox.warning(
                    self, "Pusty portfel",
                    "Dodaj co najmniej jeden instrument przed przejściem dalej."
                )
                return
            if self.stock_table.rowCount() == 0:
                QMessageBox.warning(
                    self, "Brak akcji",
                    "Portfel musi zawierać co najmniej jedną akcję."
                )
                return
            if self.bond_table.rowCount() == 0:
                QMessageBox.warning(
                    self, "Brak obligacji",
                    "Portfel musi zawierać co najmniej jedną obligację."
                )
                return
            try:
                data = self._collect_for_return_estimate()
                self.lbl_return_status.setText("⏳ Obliczanie bieżącego zwrotu portfela…")
                self.lbl_return_status.setStyleSheet("color: #555555; font-style: italic;")
                self.compute_current_return_requested.emit(data)
            except Exception:
                self.lbl_return_status.setText("")
        self._current_step = min(self._current_step + 1, 2)
        self._update_step_ui()

    def _on_estimation_params_changed(self):
        """Re-triggers return fetch when estimation window or investment horizon changes
        (only when user is past the portfolio step)."""
        if self._current_step < 1:
            return
        try:
            data = self._collect_for_return_estimate()
            self.lbl_return_status.setText("⏳ Obliczanie bieżącego zwrotu portfela…")
            self.lbl_return_status.setStyleSheet("color: #555555; font-style: italic;")
            self.compute_current_return_requested.emit(data)
        except Exception:
            pass

    def _on_prev_step(self):
        self._current_step = max(self._current_step - 1, 0)
        self._update_step_ui()

    # ── Portfolio tab ──────────────────────────────────────────────────────────

    def _build_portfolio_tab(self):
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ── Ostrzeżenie o braku aktywnych emisji ───────────────────────────────
        self.frm_no_bonds_warn = QFrame()
        self.frm_no_bonds_warn.setStyleSheet(
            "QFrame { background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; }"
        )
        _warn_row = QHBoxLayout(self.frm_no_bonds_warn)
        _warn_row.setContentsMargins(12, 8, 12, 8)
        _lbl_warn = QLabel("⚠  Brak aktywnych emisji obligacji w bazie danych. Dodaj emisje, aby móc dodawać obligacje do portfela.")
        _lbl_warn.setWordWrap(True)
        _btn_go = QPushButton("Przejdź do zbioru obligacji →")
        _btn_go.clicked.connect(self.go_bonds_requested.emit)
        _warn_row.addWidget(_lbl_warn, stretch=1)
        _warn_row.addWidget(_btn_go)
        self.frm_no_bonds_warn.setVisible(False)
        layout.addWidget(self.frm_no_bonds_warn)

        # ── Akcje ──────────────────────────────────────────────────────────────
        grp_stocks = QGroupBox("Akcje")
        vbox_s = QVBoxLayout(grp_stocks)
        toolbar_s = QHBoxLayout()
        btn_add_s = QPushButton("+ Dodaj akcję")
        btn_add_s.clicked.connect(self._open_stock_dialog)
        btn_del_s = QPushButton("− Usuń zaznaczone")
        btn_del_s.clicked.connect(self._delete_stock_rows)
        toolbar_s.addWidget(btn_add_s)
        toolbar_s.addWidget(btn_del_s)
        toolbar_s.addStretch()
        vbox_s.addLayout(toolbar_s)

        self.stock_table = QTableWidget()
        stock_cols = ["Ticker", "Ilość", "Data nabycia", "Cena nabycia (PLN)"]
        self.stock_table.setColumnCount(len(stock_cols))
        self.stock_table.setHorizontalHeaderLabels(stock_cols)
        self.stock_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stock_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.stock_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stock_table.setAlternatingRowColors(True)
        self.stock_table.setMinimumHeight(140)
        vbox_s.addWidget(self.stock_table)
        layout.addWidget(grp_stocks)

        # ── Obligacje ──────────────────────────────────────────────────────────
        grp_bonds = QGroupBox("Obligacje detaliczne (EDO / COI)")
        vbox_b = QVBoxLayout(grp_bonds)
        toolbar_b = QHBoxLayout()
        btn_add_b = QPushButton("+ Dodaj obligację")
        btn_add_b.clicked.connect(self._open_bond_selector)
        btn_del_b = QPushButton("− Usuń zaznaczone")
        btn_del_b.clicked.connect(self._delete_bond_rows)
        toolbar_b.addWidget(btn_add_b)
        toolbar_b.addWidget(btn_del_b)
        toolbar_b.addStretch()
        vbox_b.addLayout(toolbar_b)

        self.bond_table = QTableWidget()
        bond_cols = ["Symbol emisji", "Ilość", "Data emisji", "Data nabycia"]
        self.bond_table.setColumnCount(len(bond_cols))
        self.bond_table.setHorizontalHeaderLabels(bond_cols)
        self.bond_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bond_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bond_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.bond_table.setAlternatingRowColors(True)
        self.bond_table.setMinimumHeight(140)
        vbox_b.addWidget(self.bond_table)
        layout.addWidget(grp_bonds)

        # ── Dodatkowy kapitał ──────────────────────────────────────────────────
        grp_cash = QGroupBox("Dodatkowy kapitał do zainwestowania")
        cash_form = QFormLayout(grp_cash)
        cash_form.setContentsMargins(12, 8, 12, 8)
        self.edit_extra_cash = QLineEdit()
        self.edit_extra_cash.setPlaceholderText("Opcjonalnie — np. 10000")
        cash_form.addRow(
            _make_label_with_help(
                "Wolny kapitał:",
                "Dodatkowe środki pieniężne (w PLN) przeznaczone do zainwestowania\n"
                "w ramach rebalancingu portfela.\n"
                "Gotówka nie jest uwzględniana przy obliczaniu bieżących metryk\n"
                "portfela, ale wpływa na wyliczone transakcje kupna.",
            ),
            self.edit_extra_cash,
        )
        layout.addWidget(grp_cash)

        layout.addStretch()
        return scroll

    def _add_stock_row(self, data: dict = None):
        d = data or {}
        row = self.stock_table.rowCount()
        self.stock_table.insertRow(row)
        self.stock_table.setItem(row, 0, QTableWidgetItem(d.get("ticker", "")))
        self.stock_table.setItem(row, 1, QTableWidgetItem(str(d.get("quantity", ""))))
        self.stock_table.setItem(row, 2, QTableWidgetItem(d.get("date_acquired", "") or ""))
        price = d.get("price_acquired", "")
        self.stock_table.setItem(row, 3, QTableWidgetItem(str(price) if price not in ("", None) else ""))

    def _add_bond_row(self, data: dict = None):
        d = data or {}
        row = self.bond_table.rowCount()
        self.bond_table.insertRow(row)
        self.bond_table.setItem(row, 0, QTableWidgetItem(d.get("symbol_emisji") or d.get("ticker", "")))
        self.bond_table.setItem(row, 1, QTableWidgetItem(str(d.get("quantity", ""))))
        self.bond_table.setItem(row, 2, QTableWidgetItem(d.get("issue_date", "") or ""))
        self.bond_table.setItem(row, 3, QTableWidgetItem(d.get("date_acquired", "") or ""))

    def _delete_stock_rows(self):
        for idx in sorted(
            self.stock_table.selectionModel().selectedRows(),
            key=lambda x: x.row(),
            reverse=True,
        ):
            self.stock_table.removeRow(idx.row())

    def _delete_bond_rows(self):
        for idx in sorted(
            self.bond_table.selectionModel().selectedRows(),
            key=lambda x: x.row(),
            reverse=True,
        ):
            self.bond_table.removeRow(idx.row())

    def _open_stock_dialog(self):
        dlg = AddStockDialog(self)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            if data:
                self._add_stock_row(data)

    def _check_bonds_available(self):
        from datetime import date as _date
        try:
            from database import get_bond_emissions
            emissions = get_bond_emissions()
            today = _date.today()
            has_active = any(
                _date.fromisoformat(str(e["data_zakonczenia"]).strip()[:10]) > today
                for e in emissions
            )
        except Exception:
            has_active = True
        self.frm_no_bonds_warn.setVisible(not has_active)

    def _open_bond_selector(self):
        dlg = SelectBondDialog(self)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            if data:
                self._add_bond_row(data)

    # ── Parameters tab ────────────────────────────────────────────────────────

    def _build_params_tab(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        layout = QFormLayout()
        layout.setSpacing(12)
        root.addLayout(layout)

        self.combo_goal = QComboBox()
        self.combo_goal.addItem("Minimalne ryzyko (CVaR)", "min_risk")
        self.combo_goal.addItem("Maksymalny zwrot", "max_return")
        layout.addRow(
            _make_label_with_help(
                "Cel optymalizacji:",
                "Wybierz strategię optymalizacji portfela:\n"
                "• Minimalne ryzyko (CVaR) – minimalizuje warunkową wartość zagrożoną przy\n"
                "  spełnieniu minimalnej docelowej stopy zwrotu.\n"
                "• Maksymalny zwrot – maksymalizuje oczekiwany zwrot bez ograniczenia ryzyka.",
            ),
            self.combo_goal,
        )

        self.edit_goal_value = QLineEdit()
        self.edit_goal_value.setPlaceholderText("np. 8 = 8% rocznie")
        layout.addRow(
            _make_label_with_help(
                "Minimalna docelowa stopa zwrotu (%):",
                "Wymagane dla celu 'Minimalne ryzyko'.\n"
                "Roczna stopa zwrotu (w procentach), którą portfel musi co najmniej osiągnąć.\n"
                "Pole jest automatycznie wypełniane na podstawie bieżącego portfela\n"
                "po przejściu z kroku 1.",
            ),
            self.edit_goal_value,
        )
        self.lbl_return_status = QLabel("")
        self.lbl_return_status.setStyleSheet("color: #555555; font-style: italic;")
        layout.addRow("", self.lbl_return_status)
        self.combo_goal.currentIndexChanged.connect(
            lambda: self.edit_goal_value.setEnabled(
                self.combo_goal.currentData() == "min_risk"
            )
        )

        self.edit_cvar = QLineEdit()
        self.edit_cvar.setPlaceholderText("0.99")
        self.edit_cvar.setText("0.99")
        layout.addRow(
            _make_label_with_help(
                "Poziom ufności (CVaR alpha):",
                "Conditional Value at Risk (CVaR) na poziomie ufności α.\n"
                "CVaR mierzy średnią stratę w (1−α)·100% najgorszych scenariuszy.\n"
                "Np. α = 0.99 oznacza średnią stratę w 1% najgorszych przypadków.\n"
                "Typowe wartości: 0.95 lub 0.99.",
            ),
            self.edit_cvar,
        )

        self.combo_window = QComboBox()
        self.combo_window.addItems(["1Y", "2Y", "3Y", "5Y"])
        self.combo_window.setCurrentIndex(3)  # domyślnie 5Y
        layout.addRow(
            _make_label_with_help(
                "Okno estymacji:",
                "Długość okresu historycznego (w latach) używanego do estymacji\n"
                "oczekiwanych stóp zwrotu i macierzy kowariancji.\n"
                "Dłuższe okno daje stabilniejsze oszacowania, ale może być mniej\n"
                "aktualne. Krótsze okno lepiej odzwierciedla bieżące warunki rynkowe.",
            ),
            self.combo_window,
        )

        self.spin_horizon_years = QSpinBox()
        self.spin_horizon_years.setRange(1, 40)
        self.spin_horizon_years.setValue(5)
        layout.addRow(
            _make_label_with_help(
                "Horyzont inwestycyjny (lata):",
                "Planowany czas utrzymywania portfela w latach.\n"
                "Wpływa na skalowanie oczekiwanych stóp zwrotu i ryzyka\n"
                "oraz na wycenę obligacji długoterminowych.",
            ),
            self.spin_horizon_years,
        )

        self.check_planning = QCheckBox("Faza planowania (bez kosztów transakcji)")
        help_planning = _make_label_with_help(
            "",
            "W trybie fazy planowania koszty transakcji nie są uwzględniane\n"
            "podczas optymalizacji. Przydatne do symulowania docelowego portfela\n"
            "bez wpływu opłat maklerskich na wyniki.",
        )
        planning_row = QHBoxLayout()
        planning_row.setContentsMargins(0, 0, 0, 0)
        planning_row.setSpacing(4)
        planning_row.addWidget(self.check_planning)
        planning_row.addWidget(help_planning)
        planning_row.addStretch()
        planning_widget = QWidget()
        planning_widget.setLayout(planning_row)
        layout.addRow("", planning_widget)

        # Przy zmianie okna lub horyzontu przelicz bieżący zwrot portfela
        self.combo_window.currentIndexChanged.connect(self._on_estimation_params_changed)
        self.spin_horizon_years.valueChanged.connect(self._on_estimation_params_changed)

        root.addStretch()
        return w

    # ── Constraints tab ───────────────────────────────────────────────────────

    def _build_constraints_tab(self):
        w = QWidget()
        layout = QFormLayout(w)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.edit_max_weight = QLineEdit("1.0")
        layout.addRow(
            _make_label_with_help(
                "Maks. waga jednego aktywa (0–1):",
                "Maksymalny udział jednego instrumentu w portfelu.\n"
                "Wartość z zakresu 0–1, gdzie 1.0 = brak ograniczenia.\n"
                "Np. 0.3 oznacza, że żadne aktywo nie może stanowić więcej niż 30%\n"
                "wartości portfela.",
            ),
            self.edit_max_weight,
        )

        self.edit_min_trade = QLineEdit("1000")
        layout.addRow(
            _make_label_with_help(
                "Min. jednostka handlowa (PLN):",
                "Minimalna wartość zlecenia kupna lub sprzedaży w złotych.\n"
                "Transakcje o wartości poniżej tego progu nie są generowane\n"
                "podczas rebalancingu portfela.",
            ),
            self.edit_min_trade,
        )

        self.edit_tx_cost = QLineEdit("0.001")
        layout.addRow(
            _make_label_with_help(
                "Prowizja maklerska (ułamek):",
                "Opłata transakcyjna jako ułamek wartości zlecenia.\n"
                "Np. 0.001 = 0,1% wartości transakcji.\n"
                "Uwzględniana przy obliczaniu kosztów rebalancingu i przy\n"
                "porównywaniu wariantów optymalizacji.",
            ),
            self.edit_tx_cost,
        )

        return w

    # ── Default values ────────────────────────────────────────────────────────

    def _load_defaults(self):
        from pathlib import Path
        path = Path("example_input.json")
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if "additional_cash" in data and data["additional_cash"] is not None:
                self.edit_extra_cash.setText(str(data["additional_cash"]))
            p = data.get("parametry_opt", {})
            goal_key = p.get("goal_type", "min_risk")
            for i in range(self.combo_goal.count()):
                if self.combo_goal.itemData(i) == goal_key:
                    self.combo_goal.setCurrentIndex(i)
                    break
            if p.get("cvar_alpha") is not None:
                self.edit_cvar.setText(str(p["cvar_alpha"]))
            self._set_combo(self.combo_window, data.get("estimation_window", "5Y"))
            self.check_planning.setChecked(data.get("is_planning_phase", False))
            c = data.get("ustawienia_ograniczen", {})
            if "max_weight" in c:
                self.edit_max_weight.setText(str(c["max_weight"]))
            if "min_trade_unit" in c:
                self.edit_min_trade.setText(str(c["min_trade_unit"]))
            if "transaction_cost_pct" in c:
                self.edit_tx_cost.setText(str(c["transaction_cost_pct"]))
        except Exception:
            pass

    @staticmethod
    def _set_combo(combo: QComboBox, value: str):
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ── Collect & run ─────────────────────────────────────────────────────────

    def _collect(self) -> dict:
        portfolio = []

        # ── Akcje ──────────────────────────────────────────────────────────────
        for row in range(self.stock_table.rowCount()):
            def scell(col, r=row):
                item = self.stock_table.item(r, col)
                return item.text().strip() if item else ""

            ticker = scell(0)
            if not ticker:
                continue
            qty_str = scell(1)
            try:
                qty = float(qty_str)
            except ValueError:
                raise ValueError(f"Nieprawidłowa ilość dla akcji '{ticker}': '{qty_str}'")

            item_dict: dict = {"ticker": ticker, "instrument_type": "stock", "quantity": qty}
            if date_acq := scell(2):
                item_dict["date_acquired"] = date_acq
            if price_acq := scell(3):
                try:
                    item_dict["price_acquired"] = float(price_acq)
                except ValueError:
                    pass
            portfolio.append(item_dict)

        # ── Obligacje ──────────────────────────────────────────────────────────
        for row in range(self.bond_table.rowCount()):
            def bcell(col, r=row):
                item = self.bond_table.item(r, col)
                return item.text().strip() if item else ""

            symbol = bcell(0)
            if not symbol:
                continue
            qty_str = bcell(1)
            try:
                qty = float(qty_str)
            except ValueError:
                raise ValueError(f"Nieprawidłowa ilość dla obligacji '{symbol}': '{qty_str}'")

            item_dict = {"ticker": symbol, "symbol_emisji": symbol, "instrument_type": "bond", "quantity": qty}
            su = symbol.upper()
            if su.startswith("EDO"):
                item_dict["bond_type"] = "EDO"
            elif su.startswith("COI"):
                item_dict["bond_type"] = "COI"
            if issue_date := bcell(2):
                item_dict["issue_date"] = issue_date
            if date_acq := bcell(3):
                item_dict["date_acquired"] = date_acq
            portfolio.append(item_dict)

        if not portfolio:
            raise ValueError("Portfel nie może być pusty.")

        # ── Dodatkowy kapitał ──────────────────────────────────────────────────
        additional_cash = None
        if ec := self.edit_extra_cash.text().strip():
            try:
                additional_cash = float(ec)
            except ValueError:
                raise ValueError("Nieprawidłowa wartość dodatkowego kapitału PLN.")

        # Goal value (wymagane dla min_risk)
        goal_value = None
        gv = self.edit_goal_value.text().strip()
        if self.combo_goal.currentData() == "min_risk":
            if not gv:
                raise ValueError("Podaj minimalną docelową stopę zwrotu (wymagane dla celu 'Minimalne ryzyko').")
            try:
                goal_value = float(gv) / 100
            except ValueError:
                raise ValueError("Minimalna docelowa stopa zwrotu musi być liczbą (np. 8 dla 8%).")
        elif gv:
            try:
                goal_value = float(gv) / 100
            except ValueError:
                raise ValueError("Nieprawidłowa wartość celu.")

        # CVaR alpha
        cvar_alpha = None
        if cv := self.edit_cvar.text().strip():
            try:
                cvar_alpha = float(cv)
            except ValueError:
                raise ValueError("Nieprawidłowa wartość CVaR alpha.")

        def _float(widget, label):
            try:
                return float(widget.text().strip())
            except ValueError:
                raise ValueError(f"Nieprawidłowa wartość: {label}")

        max_w = _float(self.edit_max_weight, "maks. waga")
        min_trade = _float(self.edit_min_trade, "min. jednostka handlowa")
        tx_cost = _float(self.edit_tx_cost, "koszt transakcji")

        result = {
            "portfolio": portfolio,
            "additional_cash": additional_cash,
            "parametry_opt": {
                "goal_type": self.combo_goal.currentData(),
                "goal_value": goal_value,
                "cvar_alpha": cvar_alpha,
            },
            "estimation_window": self.combo_window.currentText(),
            "investment_horizon_years": self.spin_horizon_years.value(),
            "is_planning_phase": self.check_planning.isChecked(),
            "ustawienia_ograniczen": {
                "max_weight": max_w,
                "min_trade_unit": min_trade,
                "transaction_cost_pct": tx_cost,
            },
        }

        return result

    def set_current_portfolio_return(self, ret: float):
        """Pre-fills the minimum target return field with the current portfolio's expected return."""
        pct = round(ret * 100, 2)
        self.edit_goal_value.setText(f"{pct:g}")
        self.lbl_return_status.setText(f"✓ Obliczono na podstawie bieżącego portfela ({pct:g}%)")
        self.lbl_return_status.setStyleSheet("color: #007700; font-style: italic;")

    def _on_return_fetch_complete(self):
        """Called when background return fetch finishes (handles failure case)."""
        if "⏳" in self.lbl_return_status.text():
            if not self.edit_goal_value.text():
                self.lbl_return_status.setText("Nie udało się obliczyć — podaj ręcznie")
                self.lbl_return_status.setStyleSheet("color: #cc6600; font-style: italic;")
            else:
                self.lbl_return_status.setText("")

    def _collect_for_return_estimate(self) -> dict:
        """Collects minimal portfolio data needed to estimate the current return."""
        portfolio = []
        for row in range(self.stock_table.rowCount()):
            ticker_item = self.stock_table.item(row, 0)
            qty_item = self.stock_table.item(row, 1)
            if not ticker_item or not ticker_item.text().strip():
                continue
            ticker = ticker_item.text().strip()
            try:
                qty = float(qty_item.text().strip() if qty_item else "0")
            except ValueError:
                continue
            if qty <= 0:
                continue
            item: dict = {"ticker": ticker, "instrument_type": "stock", "quantity": qty}
            date_item = self.stock_table.item(row, 2)
            if date_item and date_item.text().strip():
                item["date_acquired"] = date_item.text().strip()
            price_item = self.stock_table.item(row, 3)
            if price_item and price_item.text().strip():
                try:
                    item["price_acquired"] = float(price_item.text().strip())
                except ValueError:
                    pass
            portfolio.append(item)
        for row in range(self.bond_table.rowCount()):
            sym_item = self.bond_table.item(row, 0)
            qty_item = self.bond_table.item(row, 1)
            if not sym_item or not sym_item.text().strip():
                continue
            symbol = sym_item.text().strip()
            try:
                qty = float(qty_item.text().strip() if qty_item else "0")
            except ValueError:
                continue
            if qty <= 0:
                continue
            item = {"ticker": symbol, "symbol_emisji": symbol, "instrument_type": "bond", "quantity": qty}
            su = symbol.upper()
            if su.startswith("EDO"):
                item["bond_type"] = "EDO"
            elif su.startswith("COI"):
                item["bond_type"] = "COI"
            issue_item = self.bond_table.item(row, 2)
            if issue_item and issue_item.text().strip():
                item["issue_date"] = issue_item.text().strip()
            date_acq_item = self.bond_table.item(row, 3)
            if date_acq_item and date_acq_item.text().strip():
                item["date_acquired"] = date_acq_item.text().strip()
            portfolio.append(item)
        if not portfolio:
            raise ValueError("Pusty portfel")
        return {
            "portfolio": portfolio,
            "parametry_opt": {"goal_type": "min_risk", "goal_value": 0.0},
            "estimation_window": self.combo_window.currentText(),
            "investment_horizon_years": self.spin_horizon_years.value(),
            "ustawienia_ograniczen": {
                "max_weight": 1.0,
                "min_trade_unit": 1000.0,
                "transaction_cost_pct": 0.001,
            },
        }

    def _on_run(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Zastrzeżenie prawne")
        dlg.setIcon(QMessageBox.Warning)
        dlg.setText(
            "<b>Uwaga — narzędzie o charakterze poglądowym</b>"
        )
        dlg.setInformativeText(
            "Wyniki generowane przez tę aplikację <b>nie stanowią profesjonalnej porady "
            "inwestycyjnej ani rekomendacji</b> w rozumieniu przepisów prawa.\n\n"
            "Aplikacja ma wyłącznie charakter <b>poglądowy, analityczny i symulacyjny</b>. "
            "Wszelkie decyzje inwestycyjne podejmujesz na własne ryzyko.\n\n"
            "Czy chcesz kontynuować?"
        )
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dlg.setDefaultButton(QMessageBox.No)
        if dlg.exec() != QMessageBox.Yes:
            return
        try:
            self.run_requested.emit(self._collect())
        except ValueError as exc:
            QMessageBox.warning(self, "Błąd walidacji", str(exc))


# ─── Screen 2: Results – helpers ─────────────────────────────────────────────

_METRIC_LABELS: dict[str, str] = {
    "expected_return":          "Oczekiwany zwrot",
    "volatility":               "Zmienność",
    "sharpe_ratio":             "Wskaźnik Sharpe'a",
    "estimated_execution_cost": "Koszt wykonania",
    "total_rebalancing_cost":   "Łączny koszt rebalancingu",
    "cost_impact_bps":          "Wpływ kosztów (bps)",
    "caps_auto_adjusted":       "Limity auto-korekta",
    "solver_status":            "Status solvera",
    "solver_message":           "Komunikat solvera",
}

_METRIC_TOOLTIPS: dict[str, str] = {
    "expected_return":
        "Roczna oczekiwana stopa zwrotu portfela (w %),\n"
        "obliczona jako średnioważona stopa zwrotu aktywów\n"
        "z modelu estymacji parametrów.",
    "volatility":
        "Roczna zmienność portfela (odchylenie standardowe zwrotów).\n"
        "Mierzy rozproszenie wyników wokół średniego zwrotu —\n"
        "wyższa wartość oznacza większe ryzyko.",
    "sharpe_ratio":
        "Wskaźnik Sharpe'a = (zwrot − stopa wolna od ryzyka) / zmienność.\n"
        "Mierzy nadwyżzkowy zwrot na jednostkę ryzyka.\n"
        "Wyższe wartośći są lepsze.",
    "total_rebalancing_cost":
        "Łączny szacowany koszt prowizji maklerskich\n"
        "wynikający z realizacji wszystkich transakcji rebalancingu.\n"
        "Obliczony jako suma prowizji od każdego zlecenia.",
    "cost_impact_bps":
        "Wpływ kosztów transakcyjnych wyrażony w punktach bazowych (bps).\n"
        "1 bps = 0,01% wartości portfela.\n"
        "Pozwala ocenić, jak bardzo koszty rebalancingu\n"
        "obniżają efektywną stopę zwrotu.",
}

_RESULTS_COLUMN_TOOLTIPS: dict[str, str] = {
    "Ticker": "Symbol giełdowy instrumentu.",
    "Waga": "Udział instrumentu w bieżącym portfelu (% wartości).",
    "Ilość": "Liczba posiadanych sztuk instrumentu.",
    "Optymalna waga": "Docelowy udział instrumentu po rebalancingu (% wartości portfela).",
    "Ilość docelowa": "Docelowa liczba sztuk instrumentu po zrealizowaniu transakcji rebalancingu.",
    "Operacja": "Kierunek transakcji: Kupno (zwiększenie pozycji) lub Sprzedaż (zmniejszenie).",
    "Cena est. (PLN)": "Szacowana cena instrumentu użyta do obliczenia liczby sztuk (cena ostatniej wyceny).",
    "Koszt prow. (PLN)": "Szacowany koszt prowizji maklerskiej dla tej transakcji.",
    "Scenariusz": "Nazwa scenariusza szokowego zastosowanego w stress-teście.",
    "Oczekiwany zwrot": "Roczna oczekiwana stopa zwrotu portfela w danym scenariuszu szokowym.",
    "Zmienność": "Roczna zmienność portfela (odch. std.) w danym scenariuszu szokowym.",
    "Sharpe Ratio": "Wskaźnik Sharpe'a portfela w danym scenariuszu szokowym.",
}

_CAP_LABELS: dict[str, str] = {
    "stock_cap_requested": "Maks. waga jednego aktywa (żądana)",
    "stock_cap_effective": "Maks. waga jednego aktywa (efektywna)",
    "bond_cap_requested":  "Limit obligacji (żądany)",
    "bond_cap_effective":  "Limit obligacji (efektywny)",
}
_CAP_FIELDS: set[str] = set(_CAP_LABELS)

_PCT_FIELDS: set[str] = {"expected_return", "volatility"}
_PLN_FIELDS: set[str] = {"estimated_execution_cost", "total_rebalancing_cost"}


def _metric_label(key: str) -> str:
    if key.startswith("VaR_"):
        return f"VaR {key[4:]}% (próg straty)"
    if key.startswith("CVaR_"):
        return f"CVaR {key[5:]}% (śr. strata w ogonie, ≥ VaR)"
    return _METRIC_LABELS.get(key, key.replace("_", " ").capitalize())


def _metric_tooltip(key: str) -> str:
    if key.startswith("VaR_"):
        pct = key[4:]
        return (
            f"Wartość Zagrożona (VaR) na poziomie {pct}%.\n"
            f"Maksymalna strata, która nie zostanie przekroczona\n"
            f"z prawdopodobieństwem {pct}% w danym horyzoncie."
        )
    if key.startswith("CVaR_"):
        pct = key[5:]
        return (
            f"Warunkowa Wartość Zagrożona (CVaR) na poziomie {pct}%.\n"
            f"Średnio oczekiwana strata w najgorszych (100\u2212{pct})% scenariuszy.\n"
            f"CVaR jest zawsze \u2265 VaR."
        )
    return _METRIC_TOOLTIPS.get(key, "")


def _metric_form_label(key: str) -> QWidget:
    """Returns a label+help-icon widget for a metric key, or plain QLabel if no tooltip."""
    label_text = f"{_metric_label(key)}:"
    tip = _metric_tooltip(key)
    if tip:
        return _make_label_with_help(label_text, tip)
    return QLabel(label_text)


def _fmt_metric(key: str, val) -> str:
    if isinstance(val, bool):
        return "Tak" if val else "Nie"
    if isinstance(val, float):
        if key in _PCT_FIELDS or key.startswith("VaR_") or key.startswith("CVaR_"):
            return f"{val:.2%}"
        if key == "cost_impact_bps":
            return f"{val:.1f}"
        if key in _PLN_FIELDS:
            return f"{val:,.2f} PLN"
        if key == "sharpe_ratio":
            return f"{val:.4f}"
        return f"{val:.4f}"
    return str(val)


# ─── Screen 2: Results ────────────────────────────────────────────────────────

class ResultsScreen(QWidget):
    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(10)

        # Header
        header = QHBoxLayout()
        btn_back = QPushButton("← Powrót")
        btn_back.clicked.connect(self.back_requested)
        btn_back.setFixedWidth(100)
        header.addWidget(btn_back)
        title = QLabel("Wyniki Optymalizacji")
        title.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setPointSize(18)
        f.setBold(True)
        title.setFont(f)
        header.addWidget(title, stretch=1)
        header.addSpacing(100)  # symetryczny odstęp po prawej
        root.addLayout(header)

        # Progress
        self.lbl_progress = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)   # indeterminate spinner
        root.addWidget(self.lbl_progress)
        root.addWidget(self.progress_bar)

        # Result tabs (hidden while loading)
        self.result_tabs = QTabWidget()
        self.result_tabs.setVisible(False)
        root.addWidget(self.result_tabs, stretch=1)

        # Build each tab
        self.tab_portfolio = QWidget()
        self.tab_trades = QWidget()
        self.tab_sim = QWidget()
        self.tab_stress = QWidget()
        self.tab_log = QWidget()

        for tab, label in [
            (self.tab_portfolio, "Portfel"),
            (self.tab_trades, "Transakcje"),
            (self.tab_sim, "Symulacja MC"),
            (self.tab_stress, "Stress-testy"),
            (self.tab_log, "Log"),
        ]:
            self.result_tabs.addTab(tab, label)

        log_layout = QVBoxLayout(self.tab_log)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)

        # Stałe layouty dla zakładek dynamicznych (inicjowane raz)
        QVBoxLayout(self.tab_portfolio)
        QVBoxLayout(self.tab_trades)
        QVBoxLayout(self.tab_sim)
        QVBoxLayout(self.tab_stress)

    # ── Public interface ──────────────────────────────────────────────────────

    def show_loading(self, message: str = "Trwa obliczanie…"):
        self.result_tabs.setVisible(True)
        # Zablokuj wszystkie zakładki oprócz Log podczas obliczeń
        log_idx = self.result_tabs.indexOf(self.tab_log)
        for i in range(self.result_tabs.count()):
            self.result_tabs.setTabEnabled(i, i == log_idx)
        self.result_tabs.setCurrentWidget(self.tab_log)
        self.progress_bar.setVisible(True)
        self.lbl_progress.setText(message)
        self.log_text.clear()
        self.log_text.append("[INFO] Trwa obliczanie, proszę czekać...")

    def update_progress(self, message: str):
        self.lbl_progress.setText(message)
        self.log_text.append(f"[PROGRESS] {message}")

    def append_log(self, message: str):
        self.log_text.append(message)

    def show_error(self, message: str):
        self.progress_bar.setVisible(False)
        self.lbl_progress.setText("Wystąpił błąd — szczegóły w zakładce Log.")
        self.log_text.append(f"\n[BŁĄD]\n{message}")
        self.result_tabs.setVisible(True)
        # Odblokuj wszystkie zakładki przy błędzie
        for i in range(self.result_tabs.count()):
            self.result_tabs.setTabEnabled(i, True)
        self.result_tabs.setCurrentWidget(self.tab_log)
        w = self.window()
        if hasattr(w, "stack"):
            w.stack.setCurrentIndex(1)

    def show_results(self, results: dict):
        self.progress_bar.setVisible(False)
        self.lbl_progress.setText("Zakończono pomyślnie.")
        self.result_tabs.setVisible(True)
        # Odblokuj wszystkie zakładki
        for i in range(self.result_tabs.count()):
            self.result_tabs.setTabEnabled(i, True)

        opt = results.get("optimization")
        sim = results.get("simulation")
        fills = [
            (self._fill_portfolio_tab, (results.get("current_portfolio", {}) or {}, opt)),
            (self._fill_trades_tab, (opt,)),
            (self._fill_sim_tab, (sim,)),
            (self._fill_stress_tab, (sim,)),
        ]
        for fn, args in fills:
            try:
                fn(*args)
            except Exception as exc:
                import traceback as _tb
                self.log_text.append(f"[BŁĄD renderowania {fn.__name__}]: {exc}\n{_tb.format_exc()}")
        self.result_tabs.setCurrentIndex(0)

    # ── Tab builders ──────────────────────────────────────────────────────────

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    @staticmethod
    def _make_table(rows_data: list, headers: list) -> QTableWidget:
        tbl = QTableWidget(len(rows_data), len(headers))
        tbl.setHorizontalHeaderLabels(headers)
        # Ustaw tooltips na nagłówkach kolumn
        for col, h in enumerate(headers):
            tip = _RESULTS_COLUMN_TOOLTIPS.get(h, "")
            if tip:
                tbl.horizontalHeaderItem(col).setToolTip(tip)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        for r, row in enumerate(rows_data):
            for c, val in enumerate(row):
                tbl.setItem(r, c, QTableWidgetItem(str(val)))
        return tbl

    def _fill_portfolio_tab(self, before: dict, opt):
        before = before or {}
        opt = opt or {}
        outer = self.tab_portfolio.layout()
        self._clear_layout(outer)

        # ── Wiersz z dwiema kolumnami ─────────────────────────────────────────
        columns_widget = QWidget()
        columns_row = QHBoxLayout(columns_widget)
        columns_row.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(columns_widget, stretch=1)

        # ── Lewa kolumna: Przed optymalizacją ────────────────────────────────
        left_box = QGroupBox("Przed optymalizacją")
        left = QVBoxLayout(left_box)

        value = before.get("total_value", 0.0)
        grp_bm = QGroupBox("Metryki bieżącego portfela")
        bml = QFormLayout(grp_bm)
        bml.addRow(
            _make_label_with_help(
                "Wartość portfela:",
                "Sumaryczna wartość bieżącego portfela\n"
                "(łącznie z dodatkowym kapitałem gotówkowym) w PLN."
            ),
            QLabel(f"{value:,.2f} PLN"),
        )
        for k, v in before.get("metrics", {}).items():
            bml.addRow(_metric_form_label(k), QLabel(_fmt_metric(k, v)))
        left.addWidget(grp_bm)

        bw = before.get("weights", {})
        if bw:
            qty_map = before.get("quantities", {})
            grp_bw = QGroupBox("Bieżące wagi")
            QVBoxLayout(grp_bw).addWidget(
                self._make_table(
                    [
                        (t, f"{w:.2%}", f"{qty_map.get(t, 0):g}" if t in qty_map else "")
                        for t, w in bw.items()
                    ],
                    ["Ticker", "Waga", "Ilość"]
                )
            )
            left.addWidget(grp_bw)
        left.addStretch()
        columns_row.addWidget(left_box)

        # ── Prawa kolumna: Po optymalizacji ──────────────────────────────────
        right_box = QGroupBox("Po optymalizacji")
        right = QVBoxLayout(right_box)

        if not opt:
            right.addWidget(QLabel("Optymalizacja nie powiodła się."))
        else:
            metrics_all = opt.get("metrics", {})
            budget = metrics_all.get("budget", before.get("total_value", 0.0))
            total_value_opt = metrics_all.get("total_value", budget)
            cash_remainder = metrics_all.get("cash_remainder", budget - total_value_opt)
            main_metrics = {k: v for k, v in metrics_all.items()
                            if k not in _CAP_FIELDS
                            and k not in {"caps_auto_adjusted", "solver_status", "solver_message",
                                          "estimated_execution_cost", "total_value",
                                          "budget", "cash_remainder"}}
            grp_am = QGroupBox("Metryki zoptymalizowanego portfela")
            aml = QFormLayout(grp_am)
            aml.addRow(
                _make_label_with_help(
                    "Budżet (przed zaokr.):",
                    "Całkowity kapitał przeznaczony do inwestycji\n"
                    "(wartość bieżącego portfela + wolna gotówka).\n"
                    "To samo co wartość portfela w kolumnie 'Przed'."
                ),
                QLabel(f"{budget:,.2f} PLN"),
            )
            aml.addRow(
                _make_label_with_help(
                    "Zainwestowana wartość:",
                    "Wartość portfela po rebalancingu, obliczona\n"
                    "ze zaokrąglonych ilości całkowitych sztuk.\n"
                    "Może różnić się od budżetu o niezainwestowaną resztę."
                ),
                QLabel(f"{total_value_opt:,.2f} PLN"),
            )
            if abs(cash_remainder) > 0.01:
                lbl_rem = QLabel(f"{cash_remainder:,.2f} PLN")
                lbl_rem.setStyleSheet("color: #cc6600;")
                aml.addRow(
                    _make_label_with_help(
                        "Niezainwestowana reszta:",
                        "Różnica między budżetem a zainwestowaną wartością.\n"
                        "Wynika z zaokrąglania ilości do całych sztuk\n"
                        "oraz filtra minimalnej jednostki transakcji.\n"
                        "Wartość dodatnia = gotówka pozostała w portfelu."
                    ),
                    lbl_rem,
                )
            for k, v in main_metrics.items():
                aml.addRow(_metric_form_label(k), QLabel(_fmt_metric(k, v)))
            right.addWidget(grp_am)

            weights = opt.get("weights")
            if weights is not None:
                import pandas as pd
                target_qty = opt.get("target_quantities", {})
                rows_w = [
                    (
                        str(t),
                        f"{float(w):.2%}",
                        f"{target_qty.get(t, 0):g}" if t in target_qty else "",
                    )
                    for t, w in weights.items()
                ]
                grp_aw = QGroupBox("Optymalne wagi")
                QVBoxLayout(grp_aw).addWidget(
                    self._make_table(rows_w, ["Ticker", "Optymalna waga", "Ilość docelowa"])
                )
                right.addWidget(grp_aw)
        right.addStretch()
        columns_row.addWidget(right_box)

        # ── Dół: Ograniczenia portfela (żądane / efektywne) ───────────────────
        if opt:
            metrics = opt.get("metrics", {})
            cap_fields = {k: v for k, v in metrics.items() if k in _CAP_FIELDS}
            adj = metrics.get("caps_auto_adjusted")
            if cap_fields:
                grp_caps = QGroupBox("Ograniczenia portfela (żądane / efektywne)")
                gcl = QFormLayout(grp_caps)
                for k, v in cap_fields.items():
                    tip = (
                        "Maksymalny udział jednego instrumentu w portfelu (wartość żądana przez użytkownika)."
                        if k == "stock_cap_requested" else
                        "Efektywny limit wagi, zastosowany przez optymalizator po ewentualnej auto-korekcie."
                        if k == "stock_cap_effective" else
                        "Maksymalny łączny udział obligacji w portfelu (żądany)."
                        if k == "bond_cap_requested" else
                        "Efektywny limit udziału obligacji po ewentualnej auto-korekcie."
                        if k == "bond_cap_effective" else ""
                    )
                    label_w = _make_label_with_help(f"{_CAP_LABELS[k]}:", tip) if tip else QLabel(f"{_CAP_LABELS[k]}:")
                    gcl.addRow(label_w, QLabel(f"{v:.2%}" if isinstance(v, float) else str(v)))
                if adj is not None:
                    gcl.addRow(
                        _make_label_with_help(
                            "Limity auto-korekta:",
                            "Czy optymalizator automatycznie poluzował ograniczenia wag,\n"
                            "gdy zadane limity były za wąskie (np. portfel z 3 aktywami\n"
                            "i limitem 0.2 nie może sumować się do 1.0)."
                        ),
                        QLabel("Tak" if adj else "Nie"),
                    )
                outer.addWidget(grp_caps)

    def _fill_trades_tab(self, opt):
        opt = opt or {}
        layout = self.tab_trades.layout()
        self._clear_layout(layout)

        if not opt:
            layout.addWidget(QLabel("Brak wyników optymalizacji."))
            return

        # ── Lista transakcji ─────────────────────────────────────────────────
        _ACTION = {"BUY": "Kupno", "SELL": "Sprzedaż"}
        transactions = opt.get("transactions", [])
        if not transactions:
            layout.addWidget(QLabel("Brak transakcji (faza planowania lub brak zmian)."))
            return

        rows = [
            (
                t.get("ticker", ""),
                _ACTION.get(t.get("action", ""), t.get("action", "")),
                t.get("quantity", ""),
                f"{t.get('price_est', 0):,.2f}",
                f"{t.get('est_cost', 0):,.2f}",
            )
            for t in transactions
        ]
        headers = ["Ticker", "Operacja", "Ilość", "Cena est. (PLN)", "Koszt prow. (PLN)"]
        layout.addWidget(self._make_table(rows, headers))

    def _fill_sim_tab(self, sim):
        sim = sim or {}
        layout = self.tab_sim.layout()
        self._clear_layout(layout)

        if not sim:
            layout.addWidget(QLabel("Brak wyników symulacji."))
            return

        mc = sim.get("monte_carlo", {})
        mc_metrics = mc.get("metrics", {}) if mc else {}

        grp = QGroupBox("Wyniki symulacji Monte Carlo")
        gl = QFormLayout(grp)
        for k, v in mc_metrics.items():
            gl.addRow(_metric_form_label(k), QLabel(_fmt_metric(k, v)))
        layout.addWidget(grp)
        layout.addStretch()

    def _fill_stress_tab(self, sim):
        sim = sim or {}
        layout = self.tab_stress.layout()
        self._clear_layout(layout)

        if not sim:
            layout.addWidget(QLabel("Brak wyników stress-testów."))
            return

        stress = sim.get("stress_tests", {})
        if not stress:
            layout.addWidget(QLabel("Brak wyników stress-testów."))
            return

        rows = []
        for scenario, res in stress.items():
            m = res.get("metrics", {})
            rows.append((
                scenario,
                f"{m.get('expected_return', 0):.2%}",
                f"{m.get('volatility', 0):.2%}",
                f"{m.get('sharpe_ratio', 0):.4f}",
            ))

        headers = ["Scenariusz", "Oczekiwany zwrot", "Zmienność", "Sharpe Ratio"]
        layout.addWidget(self._make_table(rows, headers))


# ─── Main Window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optymalizator Portfela")
        self.resize(1100, 760)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.input_screen = InputScreen()
        self.results_screen = ResultsScreen()
        self.bonds_screen = BondsScreen()
        self.history_screen = HistoryScreen()
        self.stack.addWidget(self.input_screen)     # index 0
        self.stack.addWidget(self.results_screen)   # index 1
        self.stack.addWidget(self.bonds_screen)     # index 2
        self.stack.addWidget(self.history_screen)   # index 3

        self.input_screen.run_requested.connect(self._start_optimization)
        self.input_screen.compute_current_return_requested.connect(self._start_current_return_fetch)
        self.input_screen.go_bonds_requested.connect(self._open_bonds_screen)
        self.results_screen.back_requested.connect(lambda: self.stack.setCurrentIndex(0))
        self.bonds_screen.back_requested.connect(lambda: self.stack.setCurrentIndex(0))
        self.bonds_screen.back_requested.connect(self.input_screen._check_bonds_available)
        self.history_screen.back_requested.connect(lambda: self.stack.setCurrentIndex(0))

        # Menu
        menu_bar = self.menuBar()
        nav_menu = menu_bar.addMenu("Nawigacja")
        act_input = nav_menu.addAction("Optymalizacja")
        act_input.triggered.connect(lambda: self.stack.setCurrentIndex(0))
        act_bonds = nav_menu.addAction("Zbiór obligacji")
        act_bonds.triggered.connect(self._open_bonds_screen)
        act_history = nav_menu.addAction("Historia analiz")
        act_history.triggered.connect(self._open_history_screen)

        # Inicjalizacja bazy danych
        from database import initialize_db
        initialize_db()
        self.input_screen._check_bonds_available()

        # Attach log handler so pipeline logs appear in the Log tab
        from logger_setup import setup_logger
        setup_logger()
        self._log_emitter = _LogEmitter()
        self._log_handler = _QTextEditHandler(self._log_emitter)
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self._log_emitter.message.connect(self.results_screen.append_log)
        logging.getLogger().addHandler(self._log_handler)

        self._worker: OptimizationWorker | None = None
        self._thread: QThread | None = None
        self._is_optimization_running: bool = False
        self._run_had_terminal_signal: bool = False
        self._last_start_ts: float = 0.0
        self._return_fetch_thread: CurrentReturnThread | None = None
        self._return_fetch_timer: QTimer | None = None
        self._return_fetch_timed_out: bool = False

    def _open_bonds_screen(self):
        self.bonds_screen._load_bonds()
        self.stack.setCurrentIndex(2)

    def _open_history_screen(self):
        self.history_screen._load_history()
        self.stack.setCurrentIndex(3)

    def _start_optimization(self, input_dict: dict):
        from models import InputData

        now = time.monotonic()
        if now - self._last_start_ts < 0.5:
            self.results_screen.append_log("[UI] Zbyt szybkie ponowne kliknięcie 'Optymalizuj' — żądanie pominięte.")
            return
        self._last_start_ts = now

        if self._is_optimization_running:
            QMessageBox.information(
                self,
                "Optymalizacja w toku",
                "Optymalizacja już trwa. Poczekaj na zakończenie bieżących obliczeń.",
            )
            return

        input_dict.pop("_strategy_id", None)

        try:
            validated = InputData.model_validate(input_dict)
        except Exception as exc:
            QMessageBox.critical(self, "Błąd walidacji danych", str(exc))
            return

        if validated.start_date is None:
            validated.start_date = date.today()

        self._is_optimization_running = True
        self._run_had_terminal_signal = False
        self.input_screen.btn_run.setEnabled(False)

        self.stack.setCurrentIndex(1)
        self.results_screen.show_loading("Inicjalizacja…")
        self.results_screen.append_log("[UI] Start optymalizacji...")
        logging.getLogger().setLevel(logging.DEBUG)

        self._thread = QThread()
        self._worker = OptimizationWorker(validated)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.results_screen.update_progress)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_optimization_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.finished.connect(self._thread.deleteLater)

        try:
            self._thread.start()
        except Exception as exc:
            self._on_optimization_finished()
            self.stack.setCurrentIndex(0)
            QMessageBox.critical(self, "Błąd uruchomienia", f"Nie udało się uruchomić optymalizacji: {exc}")

    def _on_optimization_finished(self):
        logging.getLogger().setLevel(logging.INFO)
        if not self._run_had_terminal_signal:
            self.results_screen.show_error(
                "Wątek zakończył pracę bez przekazania wyniku ani błędu. "
                "Sprawdź log aplikacji i spróbuj ponownie."
            )
        self._is_optimization_running = False
        self.input_screen.btn_run.setEnabled(True)
        self._worker = None

    def _on_worker_result(self, results: dict):
        self._run_had_terminal_signal = True
        self.results_screen.show_results(results)
        try:
            current_ret = results["current_portfolio"]["metrics"]["expected_return"]
            self.input_screen.set_current_portfolio_return(float(current_ret))
        except (KeyError, TypeError, ValueError):
            pass

    def _on_worker_error(self, message: str):
        self._run_had_terminal_signal = True
        self.results_screen.show_error(message)

    def _on_thread_finished(self):
        # Referencję do QThread zwalniamy dopiero po faktycznym zatrzymaniu wątku.
        self._thread = None

    def _start_current_return_fetch(self, portfolio_data: dict):
        if self._is_optimization_running or self._return_fetch_thread is not None:
            self.input_screen._on_return_fetch_complete()
            return
        from models import InputData
        try:
            validated = InputData.model_validate(portfolio_data)
            validated.start_date = date.today()
        except Exception as exc:
            logger.warning(f"[ReturnFetch] Walidacja InputData nie powiodła się: {exc}")
            self.input_screen._on_return_fetch_complete()
            return
        # QThread subclass – parent=self means Qt owns the C++ object,
        # so setting self._return_fetch_thread = None later is always safe.
        thread = CurrentReturnThread(validated, parent=self)
        thread.result_ready.connect(self._on_current_return_ready)
        thread.finished.connect(self._on_return_fetch_finished)
        self._return_fetch_thread = thread
        # Safety timeout: if the network call hangs, unblock the UI after 30 s
        self._return_fetch_timer = QTimer(self)
        self._return_fetch_timer.setSingleShot(True)
        self._return_fetch_timer.timeout.connect(self._on_fetch_timeout)
        self._return_fetch_timer.start(30_000)
        thread.start()

    def _on_current_return_ready(self, ret: float):
        self.input_screen.set_current_portfolio_return(ret)

    def _on_fetch_timeout(self):
        """Network call is taking too long – unblock the UI immediately.
        The thread is left running; _on_return_fetch_finished will clean up
        whenever it eventually finishes (result will be silently ignored)."""
        self._return_fetch_timer = None
        self._return_fetch_timed_out = True
        self.input_screen._on_return_fetch_complete()

    def _on_return_fetch_finished(self):
        if self._return_fetch_timer is not None:
            self._return_fetch_timer.stop()
            self._return_fetch_timer = None
        self._return_fetch_thread = None  # safe: Qt owns object via parent=self
        timed_out = self._return_fetch_timed_out
        self._return_fetch_timed_out = False
        if not timed_out:
            self.input_screen._on_return_fetch_complete()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    # Force UTF-8 on Windows console
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
