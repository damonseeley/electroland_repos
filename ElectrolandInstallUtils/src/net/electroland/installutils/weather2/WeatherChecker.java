package net.electroland.installutils.weather2;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.TimeZone;

import net.electroland.utils.ElectrolandProperties;
import uk.me.jstott.coordconv.LatitudeLongitude;
import uk.me.jstott.sun.Sun;
import uk.me.jstott.sun.Time;

public class WeatherChecker implements Runnable {

    
    private LatitudeLongitude   geo;
    private TimeZone            timeZone;
    private GregorianCalendar   gc          = new GregorianCalendar();
    private boolean             dst         = true;
    private int                 sunrisePaddingMinutes;
    private int                 sunsetPaddingMinutes;

    public WeatherChecker(ElectrolandProperties weatherConfigProps)
    {
        geo = new LatitudeLongitude(
            weatherConfigProps.getRequiredDouble("settings", "weather", "latitude"),
            weatherConfigProps.getRequiredDouble("settings", "weather", "longitude"));

        timeZone =  TimeZone.getTimeZone(weatherConfigProps.getRequired("settings", "weather", "timezone"));

        sunrisePaddingMinutes = weatherConfigProps.getDefaultInt("settings", "weather", "sunrisePadding", 0);
        sunsetPaddingMinutes = weatherConfigProps.getDefaultInt("settings", "weather", "sunsetPadding", 0);
    }

    public static void main(String args[]){
        System.out.println(new WeatherChecker(new ElectrolandProperties("weather.properties")).isDaylight());
    }

    public void addWeatherListener(WeatherListener listener)
    {
        // TODO
    }
    public void removeWeatherListener(WeatherListener listener)
    {
        // TODO
    }

    @Override
    public void run() {
        // TODO
    }

    public boolean isDaylight()
    {
        double day = gc.get(GregorianCalendar.DAY_OF_YEAR);

        Calendar now        = Calendar.getInstance();
        now.setTimeZone(timeZone);

        Calendar sunrise    = convert(Sun.sunriseTime(day, geo, timeZone, dst));
        Calendar sunset     = convert(Sun.sunsetTime( day, geo, timeZone, dst));

        sunrise.setTimeInMillis(sunrise.getTimeInMillis() + 1000 * 60 * sunrisePaddingMinutes);
        sunset.setTimeInMillis(sunset.getTimeInMillis() + 1000 * 60 * sunsetPaddingMinutes);

//        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
//        System.out.println(dateFormat.format(sunrise.getTime()));
//        System.out.println(dateFormat.format(now.getTime()));
//        System.out.println(dateFormat.format(sunset.getTime()));

        return now.after(sunrise) && now.before(sunset);
    }

    private Calendar convert(Time time){
        Calendar cal = Calendar.getInstance();

        cal.setTimeZone(timeZone);

        cal.set(cal.get(Calendar.YEAR), 
                cal.get(Calendar.MONTH), 
                cal.get(Calendar.DATE), 
                time.getHours(), 
                time.getMinutes(), 
                (int)time.getSeconds());

        return cal;
    }
}