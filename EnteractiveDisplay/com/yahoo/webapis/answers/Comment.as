/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers
{
	import com.yahoo.webapis.answers.User;
	
	/**
	 * Comment is a ValueObject for the Answers API.
	 */
	public class Comment
	{
		private var _content:String;
		private var _timestamp:Number;
		private var _author:User;
		private var _date:String;
		
		
		/**
		 * Construct a new Comment instance.
		 *
		 * @langversion ActionScript 3.0
		 * @playerversion Flash 9
		 * @author Hepp Maccoy 01/26/07
		 * @tiptext
		 */
		public function Comment()
		{
			// Do Nothing.
		}
		
		
		
		/**
		 * The content of the Comment.
		 */
		public function get content():String
		{
			return _content;
		}
		
		public function set content( value:String ):void
		{
			_content = value;
		}
		
		
		/**
		 * The User object of who posted the Comment.
		 */
		public function get author():User
		{
			return _author;
		}
		
		public function set author( value:User ):void
		{
			_author = value;
		}
		
		
		
		/**
		 * The Posted timestamp of the Comment.
		 */
		public function get timestamp():Number
		{
			return _timestamp;
		}
		
		public function set timestamp( value:Number ):void
		{
			_timestamp = value;
		}
		
		
		
		/**
		 * The Date of the Comment.
		 */
		public function get date():String
		{
			return _date;
		}
		
		public function set date( value:String ):void
		{
			_date = value;
		}
	}
}