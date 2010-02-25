/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	/**
	 * Category is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
	 
	public class Category
	{
		private var _id:String;
		private var _name:String;
		private var _url:String;
		
		/**
		 * Construct a new Category instance.
		 */
		public function Category()
		{
			// Do Nothing.
		}
		

		/**
		 * The ID of the Category.
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
		 * The Name of the Category.
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
		 * The URL of the Category at Yahoo!.
		 */
		public function get url():String
		{
			return _url;
		}
		
		public function set url( value:String ):void
		{
			_url = value;
		}
		
		public function toString():String
		{
			return name;
		}
		
	}
}