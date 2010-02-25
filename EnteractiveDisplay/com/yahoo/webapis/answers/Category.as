/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers
{
	/**
	 * Category is a ValueObject for the Answers API.
	 */
	public class Category
	{
		private var _id:String;
		private var _name:String;
		
		/**
		 * Construct a new Question instance.
		 *
		 * @langversion ActionScript 3.0
		 * @playerversion Flash 9
		 * @author Hepp Maccoy 01/04/07
		 * @tiptext
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
	}
}