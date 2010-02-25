/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.weather
{
	/**
	* Yahoo! Weather API Class. This class holds the data on current
	* and forecast weather conditions. 
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 01/27/07
	*/
	
	public class Weather extends Object
	{
		/**
		* The URL for the weather.yahoo.com page for the corresponding
		 * weather report.
		*/
		public var link:String;
		/**
		* The description of current weather report.
		*/
		public var description:String;
		/**
		* A 2-letter code for the current report language.
		*/		
		public var language:String;
		/**
		* The date and time of the current report, with UTC offset.
		*/		
		public var date:Date;
		/**
		* Time to live for the current weather report.
		*/
		public var timeToLive:Number;
		/**
		* The geographic location for the current weather report.
		*/		
		public var location:Location;
		/**
		* The units of the current weather report.
		*/	
		public var units:Units;
		/**
		* The current conditions.
		*/		
		public var current:CurrentConditions;
		/**
		* An array of forecast conditions (an item per day).
		*/		
		public var forecast:Array;
		/**
		 * The Weather class constructor.
		 * 
 		 */
		
		public function Weather() {
		location = new Location();
		units = new Units();
		current = new CurrentConditions();
		forecast = new Array();
		date = new Date();
		}
		/**
		 * A string output of the current and forecast
		 * conditions, and other Weather object properties.
         * @return The current and forecast conditions, and other weather properties, as string.
 		 */
		
		public function toString():String {
		var output:String = "---------------------------------------\n";
		output += "Description: " + this.description + "\n";
		output += "Link: " + this.link + "\n";
		output += "Language: " + this.language + "\n";
		output += "Date: " + this.date.toDateString() + "\n";
		output += "Time To Live: " + this.timeToLive + " minutes" + "\n";
		output += "Location: " + this.location.city + " " + this.location.region + ", " + this.location.country + "\n";
		output += "Latitude: " + this.location.latitude + ", Longitude: " + this.location.longitude + "\n";
		output += "Units: Temperature - " + this.units.temperature + ", Distance - " + this.units.distance + ", Pressure - " + this.units.pressure + ", Speed - " + this.units.speed + "\n";
		output += "\n";
		output += "Current Conditions\n";
		output += "------------------\n";
		output += "Description: " + this.current.description + "\n";
		output += "Temperature: " + this.current.temperature + " " + this.units.temperature + "\n";
		output += "Image URL: " + this.current.imageURL + "\n";
		output += "\n";
		output += "Wind/Chill: " + this.current.wind.chill + " " + this.units.temperature + "\n"; 
		output += "Wind/Direction: " + this.current.wind.direction + " degrees" + "\n"; 
		output += "Wind/Speed: " + this.current.wind.speed + " " + this.units.speed + "\n";
		output += "\n";
		output += "Atmosphere/Humidity: " + this.current.atmosphere.humidity + " %" + "\n"; 
		output += "Atmosphere/Visibility: " + this.current.atmosphere.visibility + " " + this.units.distance + "\n"; 
		output += "Atmosphere/Pressure: " + this.current.atmosphere.pressure + " " + this.units.pressure + "\n";
		output += "Atmosphere/Pressure is " + this.current.atmosphere.rising + "\n";
		output += "\n";
		output += "Astronomy/Sunrise: " + this.current.astronomy.sunrise.toString() + "\n";
		output += "Astronomy/Sunset: " + this.current.astronomy.sunset.toString() + "\n";
		output += "\n";
		output += "Forecast Conditions\n";
		output += "------------------\n";
		var item:ForecastConditions;
		for each(item in this.forecast) {
		output += "Date: " + item.date.toDateString() + "\n";
		output += "Temperature low: " + item.low + " " + this.units.temperature + "\n";
		output += "Temperature high: " + item.high + " " + this.units.temperature + "\n";
		output += "Description: " + item.description + "\n";
		output += "Image URL: " + item.imageURL + "\n";
		output += "\n";		
		}
				 
		return output;
		}
		
	}
}