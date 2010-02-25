/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	/**
	 * LocalSearchResult is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
	public class LocalSearchResult extends WebSearchResult
	{
		
		private var _address:String;
		private var _city:String;
		private var _state:String;
		private var _phone:String;
		
		//TODO: rethink this variable
		private var _mapResultSetURL:String;
		
		private var _mapURL:String;
		private var _businessClickURL:String;
		private var _businessURL:String;
		
		private var _rating:Rating;
		
		private var _latitude:Number;
		private var _longitude:Number;
		private var _distance:Number;
		
		
		/**
		 * Construct a new LocalSearchResult instance.
		 */
		public function LocalSearchResult()
		{
			// Do Nothing.
		}
		
		
		/**
		 * The address.
		 */
		public function get address():String
		{
			return _address;
		}
		
		public function set address( value:String ):void
		{
			_address = value;
		}
		
		/**
		 * The city.
		 */
		public function get city():String
		{
			return _city;
		}
		
		public function set city( value:String ):void
		{
			_city = value;
		}
		
		
		/**
		 * The state.
		 */
		public function get state():String
		{
			return _state;
		}
		
		public function set state( value:String ):void
		{
			_state = value;
		}
		
		/**
		 * The phone number.
		 */
		public function get phone():String
		{
			return _phone;
		}
		
		public function set phone( value:String ):void
		{
			_phone = value;
		}
		
		
		/**
		 * The map url listing all results, which this is a part of.
		 */
		public function get mapResultSetURL():String
		{
			return _mapResultSetURL;
		}
		
		public function set mapResultSetURL( value:String ):void
		{
			_mapResultSetURL = value;
		}
		
		/**
		 * The map URL.
		 */
		public function get mapURL():String
		{
			return _mapURL;
		}
		
		public function set mapURL( value:String ):void
		{
			_mapURL = value;
		}
		
		/**
		 * The business click URL.
		 */
		public function get businessClickURL():String
		{
			return _businessClickURL;
		}
		
		public function set businessClickURL( value:String ):void
		{
			_businessClickURL = value;
		}
		
		/**
		 * The business URL.
		 */
		public function get businessURL():String
		{
			return _businessURL;
		}
		
		public function set businessURL( value:String ):void
		{
			_businessURL = value;
		}
		
		
		
		/**
		 * The rating.
		 */
		public function get rating():Rating
		{
			return _rating;
		}
		
		public function set rating( value:Rating ):void
		{
			_rating = value;
		}
		
		/**
		 * The latitude.
		 */
		public function get latitude():Number
		{
			return _latitude;
		}
		
		public function set latitude( value:Number ):void
		{
			_latitude = value;
		}
		
		/**
		 * The longitude.
		 */
		public function get longitude():Number
		{
			return _longitude;
		}
		
		public function set longitude( value:Number ):void
		{
			_longitude = value;
		}
		
		/**
		 * The distance.
		 */
		public function get distance():Number
		{
			return _distance;
		}
		
		public function set distance( value:Number ):void
		{
			_distance = value;
		}
		
		
	}
}