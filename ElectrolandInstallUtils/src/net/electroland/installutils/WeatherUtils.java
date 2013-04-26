package net.electroland.installutils;

import java.net.URL;
import java.util.Calendar;
import java.util.TimeZone;

import uk.me.jstott.coordconv.LatitudeLongitude;
import uk.me.jstott.sun.Sun;
import uk.me.jstott.sun.Time;
import uk.me.jstott.util.JulianDateConverter;

import com.sun.syndication.feed.synd.SyndEntry;
import com.sun.syndication.feed.synd.SyndFeed;
import com.sun.syndication.io.SyndFeedInput;
import com.sun.syndication.io.XmlReader;

public class WeatherUtils {
	private LatitudeLongitude latlon;		// location for calculation
	private boolean dst = true;			// currently daylight savings or not
	private TimeZone timezone;				// establishes local time
	private Calendar cal;					// current date
	private double julianDate;			// julian format date
	public static final String YAHOO_WEATHER_URL = "http://weather.yahooapis.com/forecastrss?p=USCA0638&u=f";
	private URL url;

	public WeatherUtils(double latitude, double longitude){
		latlon = new LatitudeLongitude(latitude, longitude);
	    timezone = TimeZone.getTimeZone("America/Los_Angeles");
	    cal = Calendar.getInstance();
	    julianDate = JulianDateConverter.dateToJulian(cal);
	    fetchWeather(); 
	}
	
	public Time getMorningTwilight(){
		return Sun.morningCivilTwilightTime(julianDate, latlon, timezone, dst);
	}
	
	public Time getEveningTwilight(){
		return Sun.eveningCivilTwilightTime(julianDate, latlon, timezone, dst);
	}
	
	public Time getSunrise(){
		return Sun.sunriseTime(julianDate, latlon, timezone, dst);		
	}
	
	public Time getSunset(){
		return Sun.sunsetTime(julianDate, latlon, timezone, dst);
	}
	
	public void fetchWeather(){
		SyndFeedInput input = new SyndFeedInput();
	    try{
	    	url = new URL(YAHOO_WEATHER_URL);
	    	SyndFeed feed = input.build(new XmlReader(url));
	    	SyndEntry entry = (SyndEntry) feed.getEntries().get(0);
	    	System.out.println(entry);
	    } catch(Exception e){
	    	e.printStackTrace();	    	
	    }
	}
	
	public void printTimes(){
	    //System.out.println("\n\nLos Angeles, USA - " + cal.get(Calendar.DAY_OF_MONTH) + "/" + (cal.get(Calendar.MONTH)+1) + "/" + cal.get(Calendar.YEAR));
	    System.out.println("Astronomical twilight = " + Sun.morningAstronomicalTwilightTime(julianDate, latlon, timezone, dst));
	    System.out.println("Nautical twilight     = " + Sun.morningNauticalTwilightTime(julianDate, latlon, timezone, dst));
	    System.out.println("Civil twilight        = " + Sun.morningCivilTwilightTime(julianDate, latlon, timezone, dst));
	    System.out.println("Sunrise               = " + Sun.sunriseTime(julianDate, latlon, timezone, dst));
	    System.out.println("Sunset                = " + Sun.sunsetTime(julianDate, latlon, timezone, dst));
	    System.out.println("Civil twilight        = " + Sun.eveningCivilTwilightTime(julianDate, latlon, timezone, dst));
	    System.out.println("Nautical twilight     = " + Sun.eveningNauticalTwilightTime(julianDate, latlon, timezone, dst));
	    System.out.println("Astronomical twilight = " + Sun.eveningAstronomicalTwilightTime(julianDate, latlon, timezone, dst));
	}
	
	public void setDaylightSavings(boolean dst){
		this.dst = dst;
	}
	
	public void setTimeZone(String tz){
		timezone = TimeZone.getTimeZone(tz);
	}
	
	static public void main(String[] args){
		WeatherUtils weatherUtils = new WeatherUtils(34.066460, -118.308630);
		weatherUtils.printTimes();
	}

}
