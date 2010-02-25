/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.weather
{
		/**
	* Yahoo! Weather API Location Class. This class holds the data about
	* the location associated with current weather conditions. 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 01/27/07
	*/
	public class Location
	{
	/** The city where the weather is being reported.
	 */
		public var city:String;
	/** The region where the weather is being reported.
	 */
		public var region:String;
		/**
		* The country where the weather is being reported.
		*/
		public var country:String;
		/**
		* The longitude corresponding to current weather.
		*/
		public var longitude:Number;
		/**
		* The latitude corresponding to current weather.
		*/		
		public var latitude:Number;
		/**
		 * The Location constructor.
		 */		
		public function Location () {
		}
	}
}