/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.weather
{
	/**
	* Yahoo! Weather API Units Class. This class holds the data about
	* the units used in reporting weather conditions. 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 01/27/07
	*/
	public class Units
	{
		/**
		* The units used for reporting temperature.
		*/	
		public var temperature:String;
		/**
		 * The units used for reporting distance.
		 */
		public var distance:String;
		/**
		 * The units used for reporting pressure
		 */
		public var pressure:String;
/**
 * The units used for reporting speed.
 */
		public var speed:String;
/** 
 * The constant specifying the units argument for getWeather to be metric.		
 */
		public static const METRIC_UNITS:String = "metric";
/** 
 * The constant specifying the units argument for getWeather to be English.		
 */
		public static const ENGLISH_UNITS:String = "english";
		/**
		 * The Units class constructor.
		 */		
		public function Units () {
		}
	}
}