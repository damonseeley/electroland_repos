package net.electroland.installutils.weather2;

public interface WeatherListener {
    public void RegularUpdate(RegularWeatherUpdate event);
    public void Alert(RegularWeatherUpdate event);
}