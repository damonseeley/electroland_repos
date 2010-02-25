/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	/**
	 * SearchResult is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
	public class SearchResult
	{
		private var _id:String;
		private var _name:String;
		private var _index:uint;
		
		/**
		 * Construct a new SearchResult instance.
		 */
		public function SearchResult()
		{
			// Do Nothing.
		}
		
		/**
		 * The ID.
		 */
		public function get id():String
		{
			return _id;
		}
		
		public function set id( value:String ):void
		{
			_id = value;
		}
		
		/**
		 * The name of the SearchResult.
		 */
		public function get name():String
		{
			return _name;
		}
		
		public function set name( value:String ):void
		{
			_name = value;
		} 
		
		/**
		 * The index of the SearchResult.
		 */
		public function get index():uint
		{
			return _index;
		}
		
		public function set index( value:uint ):void
		{
			_index = value;
		}
		
		
		/**
		 * The label of the SearchResult.
		 */
		public function toString():String
		{
			return _name;
		}
	}
}