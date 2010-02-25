/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.weather
{
	/**
	* Yahoo! Weather API Forecast Conditions Class. This class holds the data about
	* forecast conditions. 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 01/27/07
	*/
	public class ForecastConditions
	{
		/**
		* The date for which the forecast is made.
		*/
		public var date:Date;
		/**
		 * The low temperature forecast
		 */
		public var low:Number;
		/**
		 * The high temperature forecast
		 */
		public var high:Number;
		/**
		 * Forecast description
		 */
		public var description:String;
		/**
		* URL of an image corresponding to forecast conditions.
		*/
		public var imageURL:String;
		/**
		* Condition code.
		*/
		public var code:Number;
		/**
		 * ForecastConditions class constructor
		 */		
		public function ForecastConditions () {
			date = new Date();
		}
	}
}