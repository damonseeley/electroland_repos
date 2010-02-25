/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.weather
{
	/**
	* Yahoo! Weather API Astronomy Class. This class holds the data about
	* astronomical conditions. 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 01/27/07
	*/	
	public class Astronomy
	{
		/**
		* The date and time of current date's sunrise
		*/
		public var sunrise:Date;
		/**
		 * The date and time of current date's sunset.
		 */
		public var sunset:Date;
		
		public function Astronomy() {
			sunrise = new Date();
			sunset = new Date();
			
		}
	}
}