package net.electroland.enteractive.weather;

import java.text.ParseException;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;


public class YWeatherHandler extends DefaultHandler {
	public static final float visScaler = 1.0f / 100.0f; // convert to miles
	public WeatherRecord weatherRecord = new WeatherRecord();


	public void startElement(String uri, String localName, String qName, Attributes attributes)
	throws SAXException {
		if(qName.equals("yweather:astronomy")) {
			handleAstronomyAttributes(attributes);
		} else if (qName.equals("yweather:atmosphere")) {
			handleAtmosphereAttributes(attributes);
		} else if (qName.equals("yweather:condition")) {
			handleConditionAttributes(attributes);
		} else if (qName.equals("yweather:condition")) {
			handleConditionAttributes(attributes);
		}
	}

	public void handleConditionAttributes(Attributes attributes) {
		String codeStr = attributes.getValue("code");
		String tempStr = attributes.getValue("temp");
		weatherRecord.condition = Integer.parseInt(codeStr);
		weatherRecord.outsidetemp = Integer.parseInt(tempStr);
	}
	
	public void handleAtmosphereAttributes(Attributes attributes) {
		String visStr = attributes.getValue("visibility");
		int i = Integer.parseInt(visStr);
		weatherRecord.visibility = (float) i * visScaler;
		
	}

	public void handleAstronomyAttributes(Attributes attributes) {
		try {
			weatherRecord.sunrise = parseTime(attributes.getValue("sunrise"));
		} catch (ParseException e) {
			e.printStackTrace();
		}
		try {
			weatherRecord.sunset  = parseTime(attributes.getValue("sunset"));
		} catch (ParseException e) {
			e.printStackTrace();
		}
		
	}

	/**
	 * DateFormat doesn't seem to parse correctly so have to do it manually
	 * @param s
	 * @return
	 */
	public static Calendar parseTime(String s) throws ParseException {
		Calendar c = new GregorianCalendar();
		c.setTime(new Date());
		String[] hmSplit = s.split(":");
		if(hmSplit.length != 2) throw new ParseException("Unparseable date: \"" + s + "\"", 0);
		int h = Integer.parseInt(hmSplit[0]);
		if(h==12) h = 0; // cal format uses 0 for 12
		c.set(Calendar.HOUR, h);
		String[] mam_pmSplit = hmSplit[1].split(" ");
		if(mam_pmSplit.length != 2) throw new ParseException("Unparseable date: \"" + s + "\"", 0);
		c.set(Calendar.MINUTE, Integer.parseInt(mam_pmSplit[0]));
		if(mam_pmSplit[1].equals("am")) {
			c.set(Calendar.AM_PM, Calendar.AM);
		} else {
			c.set(Calendar.AM_PM, Calendar.PM);			
		}
		return c;
		
	}
}