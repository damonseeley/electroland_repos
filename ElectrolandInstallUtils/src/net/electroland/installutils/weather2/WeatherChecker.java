package net.electroland.installutils.weather2;

import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.TimeZone;
import java.util.Vector;

import net.electroland.utils.ElectrolandProperties;

import com.luckycatlabs.sunrisesunset.SunriseSunsetCalculator;
import com.luckycatlabs.sunrisesunset.dto.Location;

public class WeatherChecker implements Runnable {

    private Vector<WeatherListener> listeners;
    private Location            location;
    private String              timeZoneName;
    private int                 sunrisePaddingMinutes;
    private int                 sunsetPaddingMinutes;

    public WeatherChecker(ElectrolandProperties weatherConfigProps)
    {
        location     = new Location(34.0522,-118.2428);
        timeZoneName =  weatherConfigProps.getRequired("settings", "weather", "timezone");

        // TODO: getDefault borked when assigned to "settings", "optional" and there was no such line.
        sunrisePaddingMinutes = weatherConfigProps.getDefaultInt("settings", "weather", "sunrisePadding", 0);
        sunsetPaddingMinutes  = weatherConfigProps.getDefaultInt("settings", "weather", "sunsetPadding", 0);
    }

    public static void main(String args[]){
        System.out.println(new WeatherChecker(
                new ElectrolandProperties("weather.properties")).isDuringDaylightHours());
    }

    public void addWeatherListener(WeatherListener listener)
    {
        if (listeners == null){
            listeners = new Vector<WeatherListener>();
        }
        listeners.add(listener);
    }
    public void removeWeatherListener(WeatherListener listener)
    {
        if (listeners == null){
            listeners.remove(listener);
        }
    }

    @Override
    public void run() {
        // TODO
    }

    public boolean isDuringDaylightHours()
    {
        Calendar gc  = GregorianCalendar.getInstance(TimeZone.getTimeZone(timeZoneName));
        SunriseSunsetCalculator calculator = new SunriseSunsetCalculator(location, timeZoneName);

        Date now     = new Date();
        Calendar sunrise = calculator.getOfficialSunriseCalendarForDate(gc);
        Calendar sunset  = calculator.getOfficialSunsetCalendarForDate(gc);

        sunrise.add(Calendar.MINUTE, sunrisePaddingMinutes);
        sunset.add(Calendar.MINUTE, sunsetPaddingMinutes);

        return now.after(sunrise.getTime()) && now.before(sunset.getTime());
    }
}