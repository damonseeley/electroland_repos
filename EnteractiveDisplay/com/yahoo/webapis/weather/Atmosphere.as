/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.weather
{
		/**
	* Yahoo! Weather API Atmosphere Class. This class holds the data about
	* atmospheric conditions. 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 01/27/07
	*/
	public class Atmosphere
	{
		/**
		* Current humidity
		*/
		public var humidity:Number;
		/**
		* Current visibility
		*/
		public var visibility:Number;
		/**
		* Current pressure
		*/
		public var pressure:Number;
		/**
		* Current pressure differential ("rising", "dropping" or "steady")
		*/
		public var rising:String;
		
		/**
		 * Atmosphere class constructor
		 */
		public function Atmosphere() {
		}
	}
}