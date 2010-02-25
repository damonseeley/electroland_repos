/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.weather
{
	import com.yahoo.webapis.weather.events.WeatherErrorEvent;
	import com.yahoo.webapis.weather.events.WeatherResultEvent;
	
	import flash.events.EventDispatcher;
	import flash.net.URLLoader;
	import flash.net.URLRequest;
	
	[Event(name="weatherLoaded", type="com.yahoo.webapis.weather.events.WeatherResultEvent")]
	[Event(name="xmlLoadingError", type="com.yahoo.webapis.weather.events.WeatherErrorEvent")]
	[Event(name="invalidLocationError", type="com.yahoo.webapis.weather.events.WeatherErrorEvent")]
	
	/**
	* Yahoo! Weather API Service Class. This class is the main entrypoint into the
	* Yahoo! Weather AS3 Library.
	*
	* @author Allen Rabinovich 01/27/07
	*/
	public class WeatherService extends EventDispatcher
	{
 		/**
 		* The container for the current weather conditions.
 		*/
 		public var weather:Weather = new Weather();
 		private var _api_URL:String;

		/**
		* WeatherService constructor.
		*/
 		public function WeatherService()
 		{
 			_api_URL = "http://weather.yahooapis.com/forecastrss";
 			weather = new Weather();
 		}

		/**
		 * The URL for the weather API. Can be changed if necessary.
		 */
		public function get api_URL():String {
			return _api_URL;
		}
		
		public function set api_URL(value:String):void {
			_api_URL = value;	
		}
		
		/**
		* This method makes a request for the current and forecast weather conditions.
		* When the weather conditions are retrieved, the current instance of WeatherService
		* dispatches a WeatherResultEvent.WEATHER_LOADED event.
		* If an error occurs, the WeatherService dispatches a WeatherResultEvent.XML_LOADING or
                * a WeatherResultEvent.INVALID_LOCATIOn event.
		* @param location Either a U.S. zip code, or a special international location code,
		* which can be obtained at http://weather.yahoo.com
		* @param units A string, either a Units.METRIC_UNITS, or a Units.ENGLISH_UNITS,
		* specifying the units to return the data in.
		*/
		public function getWeather(location:String, units:String) : void 
		{
			sendQuery(location, units);
		}
		
		
		// Private Methods ---------------------------------------------------------------
		
		/**
		 * @private
		 * Parses data into a correct string format
		 */
		private function parseDate(oldDate:String) : String {
				var newDate:String = "";
				var oldDateArray:Array = oldDate.split(" ");
				var oldHour:Number = Number(oldDateArray[4].split(":")[0]);
				var oldMin:String = oldDateArray[4].split(":")[1];
				var newHour:String;
				
				if (oldDateArray[5] == "pm") {
					oldHour = (oldHour + 12);
					if (oldHour == 24) {
						oldHour = 12;
					}
				}
				else if (oldDateArray[5] == "am" && oldHour == 12) {
					oldHour = 0;
				}
				newHour = (oldHour < 10)? "0" + String(oldHour) : String(oldHour);
				
				newDate += oldDate.split(",")[0];
				newDate += (" " + oldDateArray[2]);
				newDate += (" " + oldDateArray[1]);
				newDate += (" " + newHour + ":" + oldMin + ":00");
				newDate += (" " + timeCodeToUTC(oldDateArray[6]));
			    newDate += (" " + oldDateArray[3]);

				return newDate;
			}
		
		/**
		 * @private
		 * Returns UTC offset based on time zone code
		 * 
		 */		
		public function timeCodeToUTC(timecode:String):String {
			switch (timecode) {
				case "GMT", "UT", "UTC", "WET":  return "UTC+0000";
				case "CET": return "UTC+0100";
				case "EET": return "UTC+0200";
				case "MSK": return "UTC+0300";
				case "IRT": return "UTC+0330";
				case "SAMT": return "UTC+0400";
				case "YEKT", "TMT", "TJT": return "UTC+0500";
				case "OMST", "NOVT", "LKT": return "UTC+0600";
				case "MMT": return "UTC+0630";
				case "KRAT", "ICT", "WIT", "WAST": return "UTC+0700";
				case "IRKT", "ULAT", "CST", "CIT", "BNT": return "UTC+0800";
				case "YAKT", "JST", "KST", "EIT": return "UTC+0900";
				case "ACST": return "UTC+0930";
				case "VLAT", "SAKT", "GST": return "UTC+1000";
				case "MAGT": return "UTC+1100";
				case "IDLE", "PETT", "NZST": return "UTC+1200";
				case "WAT": return "UTC-0100";
				case "AT": return "UTC-0200";
				case "EBT": return "UTC-0300";
				case "NT": return "UTC-0330";
				case "WBT", "AST": return "UTC-0400";
				case "EST": return "UTC-0500";
				case "CST": return "UTC-0600";
				case "MST": return "UTC-0700";
				case "PST": return "UTC-0800";
				case "YST": return "UTC-0900";
				case "AHST", "CAT", "HST": return "UTC-1000";
				case "NT": return "UTC-1100";
				case "IDLW": return "UTC-1200";
			}
			return "UTC+0000";
		}
		
		/**
		* @private
		* Internal handling and loading a Weather API Call.
		*
		* @param location 		A String specifying the location
		* @param units	        A String specifying metric or english units (Acceptable values: "metric", "english", "f" or "c")
		*/

		private function sendQuery(location:String, units:String) : void
		{
			var queryURL:String = (_api_URL + "?");
			
			if (location != null) {
			queryURL += ("p="+location);
			}
			else {
			dispatchEvent(new WeatherErrorEvent(WeatherErrorEvent.INVALID_LOCATION, "Invalid location code"));
			}

			if (units != null) {
			if (units.toLowerCase() == Units.METRIC_UNITS || units.toLowerCase() == "c") {
				queryURL += "&u=c";
			}
			}
			
			var queryXML:XML;
 			var queryXMLURL:URLRequest = new URLRequest(queryURL);
			var queryLoader:URLLoader = new URLLoader(queryXMLURL);
			queryLoader.addEventListener("complete", xmlLoaded);
			queryLoader.addEventListener("ioError", xmlError);


			
	        function xmlError (evtObj:Object):void {
	        	var error:WeatherErrorEvent = new WeatherErrorEvent(WeatherErrorEvent.XML_LOADING, "XML could not be loaded");
				dispatchEvent(error);
	        }
			
			// XML loading result method
			function xmlLoaded(evtObj:Object):void
			{
				queryXML = new XML(queryLoader.data);

			if (queryXML.channel.title.indexOf("Error") == -1) {
				weather = new Weather();
				var yweather:Namespace = queryXML.namespace("yweather");
				var geo:Namespace = queryXML.namespace("geo");
				weather.link = queryXML.channel.link;
				weather.description = queryXML.channel.description;
				weather.language = queryXML.channel.language;
				weather.date.setTime(Date.parse(parseDate(queryXML.channel.lastBuildDate)));
				weather.timeToLive = queryXML.channel.ttl;
				
				weather.location.city = queryXML.channel.yweather::location.@city;
				weather.location.region = queryXML.channel.yweather::location.@region;
				weather.location.country = queryXML.channel.yweather::location.@country;
				weather.location.longitude = queryXML.channel.item.geo::lon;
				weather.location.latitude = queryXML.channel.item.geo::lat;
				
				weather.units.temperature = queryXML.channel.yweather::units.@temperature;
				weather.units.distance = queryXML.channel.yweather::units.@distance;
				weather.units.pressure = queryXML.channel.yweather::units.@pressure;
				weather.units.speed = queryXML.channel.yweather::units.@speed;
				
				weather.current.wind.chill = queryXML.channel.yweather::wind.@chill;
				weather.current.wind.direction = queryXML.channel.yweather::wind.@direction;
				weather.current.wind.speed = queryXML.channel.yweather::wind.@speed;
				
				weather.current.atmosphere.humidity = queryXML.channel.yweather::atmosphere.@humidity;
				weather.current.atmosphere.visibility = queryXML.channel.yweather::atmosphere.@visibility;
				weather.current.atmosphere.pressure = queryXML.channel.yweather::atmosphere.@pressure;
				var rising:String = queryXML.channel.yweather::atmosphere.@rising;
				switch(rising) {
					case "0": weather.current.atmosphere.rising = "steady"; break;
					case "1": weather.current.atmosphere.rising = "rising"; break;
					case "2": weather.current.atmosphere.rising = "falling"; break;
				}
				
				var dateArray:Array = String(queryXML.channel.item.yweather::condition.@date).split(" ");
				var sunrise:String = dateArray[0] + " " + dateArray[1] + " " + dateArray[2] + " " + dateArray[3] + " " + queryXML.channel.yweather::astronomy.@sunrise + " " + dateArray[6];
				var sunset:String = dateArray[0] + " " + dateArray[1] + " " + dateArray[2] + " " + dateArray[3] + " " + queryXML.channel.yweather::astronomy.@sunset + " " + dateArray[6];
				weather.current.astronomy.sunrise.setTime(Date.parse(parseDate(sunrise)));
				weather.current.astronomy.sunset.setTime(Date.parse(parseDate(sunset)));

				weather.current.description = queryXML.channel.item.yweather::condition.@text;
				weather.current.temperature = queryXML.channel.item.yweather::condition.@temp;
				weather.current.code = queryXML.channel.item.yweather::condition.@code;
                
                var description:String = String(queryXML.channel.item.description);
				description = description.substring(description.indexOf("<img"), description.indexOf((">"), description.indexOf("<img")) + 1);
				weather.current.imageURL = description.substring(description.indexOf("src") + 5, description.indexOf("\"", description.indexOf("src")+5));            

				var forecast:XMLList = queryXML.channel.item.yweather::forecast;
				var item:XML;
            	for each(item in forecast) {
                var fc:ForecastConditions = new ForecastConditions();
                var date:String = item.@day + ", " + item.@date + " " + "12:00 am " + String(queryXML.channel.item.yweather::condition.@date).split(" ")[6];
                 fc.date.setTime(Date.parse(parseDate(date)));
                fc.low = item.@low;
                fc.high = item.@high;
                fc.description = item.@text;
                fc.code = item.@code;
                fc.imageURL = weather.current.imageURL.substring(0, weather.current.imageURL.lastIndexOf("/")+1) + item.@code + ".gif";
                weather.forecast.push(fc);
            }
                var wResults:WeatherResultEvent = new WeatherResultEvent(WeatherResultEvent.WEATHER_LOADED, weather);
				dispatchEvent(wResults);
				}
			else {
				var error:WeatherErrorEvent = new WeatherErrorEvent(WeatherErrorEvent.INVALID_LOCATION, "Location not found");
				dispatchEvent(error);
			}
			}
		}
		
	}
} 