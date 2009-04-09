package net.electroland.noho.weather;

import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.SAXException;

public class TempParser {
	public static final String WEATHERGOOSE_URL = "http://elnoho.dyndns.org/data.xml";
	URL url;

	SAXParserFactory parserFactory;

	public TempParser() {
		parserFactory = SAXParserFactory.newInstance();
		try {
			url = new URL(WEATHERGOOSE_URL);
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}			
	}

	public WeatherRecord fetch() throws ParserConfigurationException, SAXException, IOException {
		InputStream is = url.openStream();
		SAXParser parser = parserFactory.newSAXParser();
		TempHandler th = new TempHandler();
		parser.parse(is, th);
		return th.artboxTempRecord;
	}
	
	public static void main(String args[]) throws ParserConfigurationException, SAXException, IOException {
		//TempParser ywp = new TempParser();
		//ywp.fetch();
	}
}
