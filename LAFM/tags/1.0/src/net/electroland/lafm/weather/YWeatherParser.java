package net.electroland.lafm.weather;

import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.SAXException;

public class YWeatherParser {
	public static final String YAHOO_WEATHER_URL = "http://weather.yahooapis.com/forecastrss?p=USCA0638&u=f";
	URL url;

	SAXParserFactory parserFactory;

	public YWeatherParser() {
		parserFactory = SAXParserFactory.newInstance();
		try {
			url = new URL(YAHOO_WEATHER_URL);
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}			
	}

	public WeatherRecord fetch() throws ParserConfigurationException, SAXException, IOException {
		InputStream is = url.openStream();
		SAXParser parser = parserFactory.newSAXParser();
		YWeatherHandler ywh = new YWeatherHandler();
		parser.parse(is, ywh);
		return ywh.weatherRecord;
	}
	
	public static void main(String args[]) throws ParserConfigurationException, SAXException, IOException {
		YWeatherParser ywp = new YWeatherParser();
		ywp.fetch();
	}
}
