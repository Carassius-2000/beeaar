"""Bearing Vibration Prediction Information System"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
import socket
import wx
import wx.adv
from wx.lib import buttons
import psycopg2 as pspg2
from psycopg2 import pool
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP
import ssl
import matplotlib
import pandas as pd
import joblib
from matplotlib.backends.backend_wxagg import (
    NavigationToolbar2WxAgg as NavigationToolbar,
    FigureCanvasWxAgg as FigureCanvas)
from matplotlib.figure import Figure
import seaborn as sns
sns.set_theme()
matplotlib.use('WXAgg')

BEARING_LIST: List[str] = ['Первый подшипник',
                           'Второй подшипник',
                           'Третий подшипник']
MAX_BEARINGS_VIBRATION: Dict[int, int] = {0: 96, 1: 33, 2: 52}
BACKGROUND_COLOR: str = '#ffe2b0'
BUTTON_COLOR: str = '#eab0bb'
TEXT_COLOR: str = '#000d35'


class AuthorizationWindow(wx.Frame):
    """Window that allows user to enter Information System."""

    def __init__(self):
        """Create Authorization Window.

        Attributes:
            login_edit (wx.TextCtrl): Edit that contains user's login.
            password_edit (wx.TextCtrl): Edit that contains user's password.
        """
        super().__init__(parent=None,
                         title='Вход в систему',
                         size=(350, 200),
                         style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU
                         | wx.CAPTION | wx.CLOSE_BOX)
        self.Center()

        panel = wx.Panel(self)
        panel.SetFont(APP_FONT)
        panel.SetBackgroundColour(BACKGROUND_COLOR)
        ico = wx.Icon('forecast.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(ico)
        # create FlexGridSizer that contains login and password edits
        # with their respective labels
        flex_grid_sizer = wx.FlexGridSizer(2, 2, 10, 10)

        login_label = wx.StaticText(panel, label='Логин')
        login_label.SetForegroundColour(TEXT_COLOR)
        self.login_edit = wx.TextCtrl(panel,
                                      size=(230, 30))

        password_label = wx.StaticText(panel, label='Пароль')
        password_label.SetForegroundColour(TEXT_COLOR)
        self.password_edit = wx.TextCtrl(panel,
                                         size=(230, 30),
                                         style=wx.TE_PASSWORD)

        flex_grid_sizer.AddMany([(login_label),
                                 (self.login_edit, wx.ID_ANY, wx.EXPAND),
                                 (password_label),
                                 (self.password_edit, wx.ID_ANY, wx.EXPAND)])
        # add FlexGridSizer to BoxSizer
        box_sizer = wx.BoxSizer(wx.VERTICAL)
        box_sizer.Add(flex_grid_sizer, flag=wx.EXPAND | wx.ALL, border=10)

        enter_button = buttons.GenButton(panel, label='Войти')
        enter_button.SetForegroundColour(TEXT_COLOR)
        enter_button.SetBackgroundColour(BUTTON_COLOR)
        box_sizer.Add(enter_button,
                      flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        enter_button.Bind(wx.EVT_BUTTON, self.on_enter_button_click)
        panel.SetSizer(box_sizer)

        self.Bind(wx.EVT_CLOSE, self.on_close_window)

    def on_close_window(self, event) -> None:
        """Close Authorization Window."""
        question: str = 'Вы действительно хотите выйти из приложения?'
        dialog_message = wx.MessageDialog(self,
                                          question,
                                          ' ',
                                          wx.YES_NO | wx.YES_DEFAULT
                                          | wx.ICON_WARNING)
        result: int = dialog_message.ShowModal()
        if result == wx.ID_YES:
            self.Destroy()
        else:
            event.Veto()

    def get_connection_pool(self, username: str, password: str):
        """Get PostgreSQL database connection pool.

        Args:
            username (str): PostgreSQL user's name.
            password (str): PostgreSQL user's password.

        Returns:
            PostgreSQL connection pool.

        Raises:
            psycopg2.OperationalError: If username or password is invalid.
        """
        try:
            connection_pool = pool.SimpleConnectionPool(
                1, 20,
                user=username,
                password=password,
                database='bearing_db')
        except pspg2.OperationalError:
            error_text: str = 'Введен неверный логин или пароль.'
            dialog_message = wx.MessageDialog(self,
                                              error_text,
                                              ' ',
                                              wx.OK | wx.ICON_ERROR)
            dialog_message.ShowModal()
            return
        else:
            return connection_pool

    def on_enter_button_click(self, event) -> None:
        """Enter Information System."""
        login: str = self.login_edit.GetValue()
        password: str = self.password_edit.GetValue()
        if check_internet_connection():
            connection_pool = self.get_connection_pool(login, password)
            if connection_pool:
                self.Destroy()
                main_frame = MainWindow(connection_pool=connection_pool)
                main_frame.Show()


class MainWindow(wx.Frame):
    """App main window."""

    def __init__(self, connection_pool=None):
        """Create Main Window.

        Attributes:
            connection_pool (psycopg2.pool.SimpleConnectionPool):\
                PostgreSQL connection pool.
            bearing_type (int): User's bearing choice.
            predictor_matrix (pd.DataFrame):\
                DataFrame that stores input data for models.
            predictions (pd.DataFrame):\
                DataFrame that stores fitted values with confidence coridor.
        """
        super().__init__(parent=None,
                         title='Главное окно',
                         size=(350, 250),
                         style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU
                         | wx.CAPTION | wx.CLOSE_BOX)
        self.Center()

        self.connection_pool = connection_pool

        panel = wx.Panel(self)
        panel.SetFont(APP_FONT)
        panel.SetBackgroundColour(BACKGROUND_COLOR)
        ico = wx.Icon('forecast.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(ico)

        box_sizer = wx.BoxSizer(wx.VERTICAL)

        select_button = buttons.GenButton(panel, label='Выбрать дату прогноза')
        select_button.SetForegroundColour(TEXT_COLOR)
        select_button.SetBackgroundColour(BUTTON_COLOR)
        box_sizer.Add(select_button,
                      flag=wx.EXPAND |
                      wx.LEFT | wx.RIGHT | wx.BOTTOM,
                      border=10)
        select_button.Bind(wx.EVT_BUTTON, self.on_select_button_click)

        self.bearing_type: int = -1
        self.predictor_matrix = None
        self.predictions = None

        self.visualization_button = buttons.GenButton(
            panel, label='Визуализация процесса')
        self.visualization_button.Enable(False)
        self.visualization_button.SetForegroundColour(TEXT_COLOR)
        self.visualization_button.SetBackgroundColour(BUTTON_COLOR)
        box_sizer.Add(self.visualization_button,
                      flag=wx.EXPAND |
                      wx.LEFT | wx.RIGHT | wx.BOTTOM,
                      border=10)
        self.visualization_button.Bind(wx.EVT_BUTTON,
                                       self.on_visualization_button_click)

        send_message_button = buttons.GenButton(
            panel, label='Отправить сообщение')
        send_message_button.SetForegroundColour(TEXT_COLOR)
        send_message_button.SetBackgroundColour(BUTTON_COLOR)
        box_sizer.Add(send_message_button,
                      flag=wx.EXPAND
                      | wx.LEFT | wx.RIGHT | wx.BOTTOM,
                      border=10)
        send_message_button.Bind(
            wx.EVT_BUTTON, self.on_send_message_button_click)

        self.save_prediction_button = buttons.GenButton(
            panel, label='Записать прогнозы')
        self.save_prediction_button.Enable(False)
        self.save_prediction_button.SetForegroundColour(TEXT_COLOR)
        self.save_prediction_button.SetBackgroundColour(BUTTON_COLOR)
        box_sizer.Add(self.save_prediction_button,
                      flag=wx.EXPAND |
                      wx.LEFT | wx.RIGHT | wx.BOTTOM,
                      border=10)
        self.save_prediction_button.Bind(
            wx.EVT_BUTTON, self.on_save_prediction_button_click)

        panel.SetSizer(box_sizer)

        self.Bind(wx.EVT_CLOSE, self.on_close_window)

    def on_close_window(self, event) -> None:
        """Close Authorization Window."""
        question: str = 'Вы действительно хотите выйти из приложения?'
        dialog_message = wx.MessageDialog(self,
                                          question,
                                          ' ',
                                          wx.YES_NO | wx.YES_DEFAULT
                                          | wx.ICON_WARNING)
        result = dialog_message.ShowModal()

        if result == wx.ID_YES:
            self.connection_pool.closeall
            self.Destroy()
        else:
            event.Veto()

    def on_select_button_click(self, event) -> None:
        """Open Select Data Window."""
        # bearing_type: int = self.bearing_choice.GetCurrentSelection()
        with SelectDataWindow(self,
                              self.connection_pool) as select_data_dialog:
            select_data_dialog.ShowModal()
        if self.predictor_matrix is not None:
            # Enable buttons
            self.visualization_button.Enable(True)
            self.save_prediction_button.Enable(True)
            # Make predictions
            if self.bearing_type == 0:
                model = joblib.load(r'models\final_model_1st.model')
            elif self.bearing_type == 1:
                model = joblib.load(r'models\final_model_2st.model')
            elif self.bearing_type == 2:
                model = joblib.load(r'models\final_model_3st.model')
            forecast_values = model.predict(self.predictor_matrix)

            min_forecast_values, max_forecast_values = self.\
                prediction_intervals(forecast_values)
            self.predictions = pd.DataFrame(
                {'date': self.predictor_matrix.index,
                 'value': forecast_values,
                 'max_value': max_forecast_values,
                 'min_value': min_forecast_values})

    def prediction_intervals(self, y_r: np.ndarray) -> Tuple[
            np.ndarray, np.ndarray]:
        '''Prediction interval'''
        std = y_r.std()
        koef = 2.1701
        yr_min = y_r - round(koef * std, 8)
        yr_max = y_r + round(koef * std, 8)
        return yr_min, yr_max

    def on_visualization_button_click(self, event) -> None:
        """Open Plot Window."""
        with PlotWindow(self,
                        self.predictions,
                        self.bearing_type) as plot_window_dialog:
            plot_window_dialog.ShowModal()

    def on_send_message_button_click(self, event) -> None:
        """Open Send Message Window."""
        with SendMessageWindow(self) as send_message_dialog:
            send_message_dialog.ShowModal()

    def on_save_prediction_button_click(self, event) -> None:
        """Save predictions to database."""
        if check_internet_connection():
            connection = self.connection_pool.getconn()
            with connection.cursor() as cursor:
                query: str = "CALL insert_predictions(%s, %s, %s);"
                current_date = datetime.today()
                report_date: str = current_date.strftime("%Y-%m-%d %H:%M:%S")
                prediction = self.predictions.to_json(orient='records',
                                                      date_format='iso')
                parametrs: List[Any] = [
                    BEARING_LIST[self.bearing_type], report_date, prediction]
                cursor.execute(query, parametrs)
                connection.commit()
            information_text: str = 'Прогнозы успешно загружены'
            information_message = wx.MessageDialog(
                None,
                information_text,
                ' ',
                wx.OK | wx.ICON_INFORMATION)
            information_message.ShowModal()
            self.connection_pool.putconn(connection)


class SelectDataWindow(wx.Dialog):
    """Window that allows to choose prediction date."""

    def __init__(self, parent, connection_pool):
        """Create Select Data Window.

        Attributes:
            connection_pool (psycopg2.pool.SimpleConnectionPool):\
                PostgreSQL connection pool.
            bearing (str): User's bearing choice.
            parent : Parent window reference.
            date_begin_edit (wx.adv.DatePickerCtrl):\
                Control that contains first prediction date.
            date_end_edit (wx.adv.DatePickerCtrl):\
                Control that contains second prediction date.
        """
        super().__init__(parent=parent,
                         title='Выбрать дату прогноза',
                         size=(380, 250),
                         style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU
                         | wx.CAPTION | wx.CLOSE_BOX)
        self.Center()
        self.connection_pool = connection_pool
        self.parent = parent

        panel = wx.Panel(self)
        panel.SetFont(APP_FONT)
        panel.SetBackgroundColour(BACKGROUND_COLOR)

        flex_grid_sizer = wx.FlexGridSizer(3, 2, 10, 10)

        bearing_label = wx.StaticText(panel, label='Для')
        bearing_label.SetForegroundColour(TEXT_COLOR)
        self.bearing_choice = wx.Choice(panel, choices=BEARING_LIST)
        self.bearing_choice.SetSelection(0)

        date_begin_label = wx.StaticText(panel, label='C')
        date_begin_label.SetForegroundColour(TEXT_COLOR)
        self.date_begin_edit = wx.adv.DatePickerCtrl(panel,
                                                     style=wx.adv.DP_DROPDOWN,
                                                     size=(300, 30))

        date_end_label = wx.StaticText(panel, label='По')
        date_end_label.SetForegroundColour(TEXT_COLOR)
        self.date_end_edit = wx.adv.DatePickerCtrl(panel,
                                                   style=wx.adv.DP_DROPDOWN,
                                                   size=(300, 30))

        flex_grid_sizer.AddMany([(bearing_label),
                                 (self.bearing_choice, wx.ID_ANY, wx.EXPAND),
                                 (date_begin_label),
                                 (self.date_begin_edit, wx.ID_ANY, wx.EXPAND),
                                 (date_end_label),
                                 (self.date_end_edit, wx.ID_ANY, wx.EXPAND)])

        box_sizer = wx.BoxSizer(wx.VERTICAL)
        box_sizer.Add(flex_grid_sizer, flag=wx.EXPAND | wx.ALL, border=10)

        enter_button = buttons.GenButton(panel, label='Спрогнозировать')
        enter_button.SetForegroundColour(TEXT_COLOR)
        enter_button.SetBackgroundColour(BUTTON_COLOR)
        box_sizer.Add(enter_button,
                      flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)
        enter_button.Bind(wx.EVT_BUTTON, self.on_enter_button_click)
        panel.SetSizer(box_sizer)

    def on_enter_button_click(self, event) -> None:
        """Select data from DB and send it to Main Window."""
        date_begin: str = str(self.date_begin_edit.GetValue()).split()[1]
        date_end: str = str(self.date_end_edit.GetValue()).split()[1]

        # formatting date to PostgreSQL timestamp type
        date_begin = datetime(*list(map(int, date_begin.split('.')))[::-1])
        date_end = datetime(*list(map(int, date_end.split('.')))[::-1])

        bearing: int = self.bearing_choice.GetCurrentSelection()

        if self.check_date(date_begin, date_end):
            parameters: Tuple[str] = (date_begin, date_end)
            if bearing == 0:
                query: str = "SELECT * FROM X1 \
                    WHERE date_time >= %s AND date_time < %s"
                column_count: int = 16
            elif bearing == 1:
                query: str = "SELECT * FROM X2 \
                    WHERE date_time >= %s AND date_time < %s"
                column_count: int = 18
            elif bearing == 2:
                query: str = "SELECT * FROM X3 \
                    WHERE date_time >= %s AND date_time < %s"
                column_count: int = 18

            connection = self.connection_pool.getconn()
            with connection.cursor() as cursor:
                cursor.execute(query, parameters)
                predictor_matrix = cursor.fetchall()
                predictor_matrix = pd.DataFrame(
                    data=predictor_matrix,
                    columns=['col ' + str(i)
                             for i in range(column_count)])
                if predictor_matrix.empty:
                    error_text: str = 'Нет данных на эти даты.'
                    error_message = wx.MessageDialog(None,
                                                     error_text,
                                                     ' ',
                                                     wx.OK | wx.ICON_ERROR)
                    error_message.ShowModal()
                else:
                    predictor_matrix.index = predictor_matrix.iloc[:, 0]
                    predictor_matrix = predictor_matrix.drop(
                        columns=['col 0'], axis=1)

                    query_last_date = predictor_matrix.index[-1]\
                        .to_pydatetime()
                    if date_end - timedelta(minutes=10) > query_last_date:
                        warning_text: str = f'Последние данные есть\
 за {query_last_date}.'
                        warning_message = wx.MessageDialog(
                            None,
                            warning_text,
                            ' ',
                            wx.OK | wx.ICON_INFORMATION)
                        warning_message.ShowModal()
                    elif date_end - timedelta(minutes=10) == query_last_date:
                        information_text: str = 'Данные успешно получены'
                        information_message = wx.MessageDialog(
                            None,
                            information_text,
                            ' ',
                            wx.OK | wx.ICON_INFORMATION)
                        information_message.ShowModal()

                    if bearing == 0:
                        scaler = joblib.load(r'scalers\scaler1st.model')
                    elif bearing == 1:
                        scaler = joblib.load(r'scalers\scaler2st.model')
                    elif bearing == 2:
                        scaler = joblib.load(r'scalers\scaler3st.model')
                predictor_matrix = pd.DataFrame(
                    data=scaler.transform(predictor_matrix),
                    columns=predictor_matrix.columns,
                    index=predictor_matrix.index)
                self.parent.bearing_type = bearing
                self.parent.predictor_matrix = predictor_matrix
                self.connection_pool.putconn(connection)

    def check_date(self, date_begin, date_end) -> bool:
        """Check date for validity.

        Args:
            date_begin (datetime.datetime): First prediction date.
            date_end (datetime.datetime): Second prediction date.

        Returns:
            bool: True if all date is valid, False in other cases.
        """
        if date_begin == date_end:
            error_text: str = 'Даты начала и конца прогноза не могут быть равными.\
            \nЕсли хотите сделать прогноз на 24 часа,\
            \nто укажите второй датой следующий день.'
            error_message = wx.MessageDialog(None,
                                             error_text,
                                             ' ',
                                             wx.OK | wx.ICON_ERROR)
            error_message.ShowModal()
            return False

        elif date_begin > date_end:
            error_text: str = 'Дата начала прогноза не может быть раньше\
 даты окончания прогноза.'
            error_message = wx.MessageDialog(None,
                                             error_text,
                                             ' ',
                                             wx.OK | wx.ICON_ERROR)
            error_message.ShowModal()
            return False
        else:
            return True


class CanvasPanel(wx.Panel):
    """Panel that contains canvas with plot on it."""

    def __init__(self, parent):
        """Create Canvas Panel.

        Args:
            parent: Parent class reference.

        Attributes:
            axes (matplotlib.axes._subplots.AxesSubplot): \
                Axes that contain plots.
        """
        super().__init__(parent)

        figure = Figure()
        self.axes = figure.add_subplot(111)

        box_sizer = wx.BoxSizer(wx.VERTICAL)

        canvas = FigureCanvas(self, -1, figure)
        box_sizer.Add(canvas, 1, wx.LEFT | wx.TOP | wx.GROW)

        toolbar = NavigationToolbar(canvas)
        toolbar.Realize()
        box_sizer.Add(toolbar, 0, wx.LEFT | wx.EXPAND)

        self.SetSizer(box_sizer)
        self.Fit()

    def get_outliers(self,
                     predictions,
                     bearing_type: int) -> Any:
        """Get outliers DataFrame.

        Args:
            predictions (pd.core.frame.DataFrame):\
            DataFrame that contain fitted values with prediction interval.
            bearing_type (int): Need to determine the limit value of vibration.

        Returns:
            pd.core.frame.DataFrame: DataFrame that contain outliers.
        """
        condition: bool = predictions['value'] > MAX_BEARINGS_VIBRATION[
            bearing_type]
        outliers = pd.DataFrame({'value': predictions[condition].value,
                                 'date': predictions[condition].date})
        return outliers

    def show_plot(self,
                  predictions: Any,
                  bearing_type: int) -> None:
        """Show plot with predictions.

        Args:
            predictions (pd.core.frame.DataFrame):\
            DataFrame that contain fitted values with prediction interval.
            bearing_type (int): Need to determine the limit value of vibration.
        """
        bearing_name: str = BEARING_LIST[bearing_type]

        self.axes.plot(predictions['value'],
                       label='Прогнозные значения', color='blue')
        self.axes.plot(predictions['min_value'],
                       linestyle='--', color='cyan',
                       label='Минимальные прогнозные значения')
        self.axes.plot(predictions['max_value'],
                       linestyle='--', color='orange',
                       label='Максимальные прогнозные значения')

        outliers = self.get_outliers(predictions,
                                     bearing_type=bearing_type)
        if not outliers.empty:
            self.axes.plot(outliers.value,
                           linestyle='', marker='o',
                           color='red', label='Аномальные значения')
        self.axes.set_title(f'{bearing_name}', fontsize=12)
        self.axes.set_xlabel('Время (10 мин)', fontsize=12)
        self.axes.set_ylabel('Вибрация (мкм)', fontsize=12)
        self.axes.legend(loc='best', shadow=True, fontsize=12)


class PlotWindow(wx.Dialog):
    """Window that shows bearing vibration plot."""

    def __init__(self, parent,
                 predictions,
                 bearing_type: int):
        """Create Plot Window.

        Args:
            parent: Parent class reference.

        Attributes:
            panel (CanvasPanel): Panel that contains canvas with plot.
        """
        super().__init__(parent=parent,
                         title='Окно визуализации',
                         size=(900, 900),
                         style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU
                         | wx.CAPTION | wx.CLOSE_BOX
                         | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        self.Center()

        self.panel = CanvasPanel(self)

        self.panel.show_plot(predictions, bearing_type)


class SendMessageWindow(wx.Dialog):
    """Window that allows user to send message to engineer."""

    def __init__(self, parent=None):
        """Create Send Message Window.

        Args:
            parent: Parent class reference.

        Attributes:
            bearing_choice (wx.Choice): Choice that contains bearing types.
            date_edit (wx.adv.DatePickerCtrl):\
            Edit that contains user's password.
        """
        super().__init__(parent=parent,
                         title='Отправка сообщения',
                         size=(360, 200),
                         style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU
                         | wx.CAPTION | wx.CLOSE_BOX)
        self.Center()

        panel = wx.Panel(self)
        panel.SetFont(APP_FONT)
        panel.SetBackgroundColour(BACKGROUND_COLOR)

        flex_grid_sizer = wx.FlexGridSizer(2, 2, 10, 10)

        change_bearing_label = wx.StaticText(panel, label='Заменить')
        change_bearing_label.SetForegroundColour(TEXT_COLOR)
        self.bearing_choice = wx.Choice(panel, choices=BEARING_LIST)
        self.bearing_choice.SetSelection(0)

        date_label = wx.StaticText(panel, label='До')
        date_label.SetForegroundColour(TEXT_COLOR)
        self.date_edit = wx.adv.DatePickerCtrl(panel,
                                               style=wx.adv.DP_DROPDOWN,
                                               size=(200, 30))

        flex_grid_sizer.AddMany([(change_bearing_label),
                                 (self.bearing_choice,
                                  wx.ID_ANY, wx.EXPAND),
                                 (date_label),
                                 (self.date_edit, wx.ID_ANY, wx.EXPAND)])

        box_sizer = wx.BoxSizer(wx.VERTICAL)
        box_sizer.Add(flex_grid_sizer, flag=wx.EXPAND | wx.ALL, border=10)

        enter_button = buttons.GenButton(panel, label='Отправить')
        enter_button.SetForegroundColour(TEXT_COLOR)
        enter_button.SetBackgroundColour(BUTTON_COLOR)
        box_sizer.Add(enter_button,
                      flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)
        enter_button.Bind(wx.EVT_BUTTON, self.on_enter_button_click)

        panel.SetSizer(box_sizer)

    def on_enter_button_click(self, event) -> None:
        """Send message and show message dialog."""
        bearing_type: str = BEARING_LIST[
            self.bearing_choice.GetCurrentSelection()]
        date: str = str(self.date_edit.GetValue()).split()[1]
        if check_internet_connection():
            self.send_mail(bearing_type, date)
            dialog_text: str = 'Сообщение успешно отправлено'
            dialog_message = wx.MessageDialog(self,
                                              dialog_text,
                                              ' ',
                                              wx.OK | wx.ICON_INFORMATION)
            dialog_message.ShowModal()

    def send_mail(self, bearing: int, date: str) -> None:
        """Send e-mail using Simple Mail Transfer Protocol.

        Args:
            bearing (int): Bearing type that need to be replaced.
            date (str): Date by which the bearing must be replaced.
        """
        SMTP_SERVER: str = "smtp.gmail.com"
        PORT: int = 587

        with SMTP(SMTP_SERVER, PORT) as server:
            # Secure the connection
            server.starttls(context=ssl.create_default_context())

            message = self.create_mail(bearing, date)
            SENDER_MAIL: str = message['From']
            with open('pass.txt', 'r') as password_file:
                PASSWORD: str = password_file.readline()
            server.login(SENDER_MAIL, PASSWORD)

            RECEIVER_MAIL: str = message['To']
            server.sendmail(SENDER_MAIL, RECEIVER_MAIL, message.as_string())

    def create_mail(self, bearing: int, date: str):
        """Create mail with subject and relevant text.

        Args:
            bearing (int): Bearing type that need to be replaced.
            date (str): Date by which the bearing must be replaced.

        Returns:
            MIMEMultipart: Сlass that contains information about e-mail.
        """
        message = MIMEMultipart()
        message['Subject'] = 'Замена подшипника'
        message['From'] = 'skaratsev@gmail.com'
        message['To'] = 'DataScienceColab1337@gmail.com'

        text: str = f'{bearing} необходимо заменить до {date}'
        body_text = MIMEText(text, 'plain')
        message.attach(body_text)

        return message


def internet_connection_fail() -> None:
    """Show internet connection error dialog."""
    error_text: str = 'Проверьте подключение к интернету.'
    error_message = wx.MessageDialog(None,
                                     error_text,
                                     ' ',
                                     wx.OK | wx.ICON_ERROR)
    error_message.ShowModal()


def check_internet_connection() -> bool:
    """Try to ping www.yandex.ru.

    Returns:
        bool: True if the attempt was successful.
    """
    try:
        socket.create_connection(("www.yandex.ru", 80))
    except OSError:
        internet_connection_fail()
    else:
        return True


if __name__ == '__main__':
    app = wx.App()
    # get OS default font
    APP_FONT = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
    APP_FONT.SetPointSize(12)

    authorization_frame = AuthorizationWindow()
    authorization_frame.Show()
    
    # main_frame = MainWindow()
    # main_frame.Show()
    app.MainLoop()
