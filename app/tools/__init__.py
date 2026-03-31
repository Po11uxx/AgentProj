from app.tools.calendar_tool import CalendarTool
from app.tools.map_tool import MapTool
from app.tools.python_executor import PythonExecutorTool
from app.tools.search_tool import SearchTool
from app.tools.transport_tool import TransportTool
from app.tools.weather_tool import WeatherTool


def get_toolkit() -> dict[str, object]:
    return {
        "weather": WeatherTool(),
        "search": SearchTool(),
        "map": MapTool(),
        "calendar": CalendarTool(),
        "transport": TransportTool(),
        "python_executor": PythonExecutorTool(),
    }
