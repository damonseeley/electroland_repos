/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers
{
	/**
	 * User is a ValueObject for the Answers API.
	 */
	public class User
	{
		private var _userId:String;
		private var _userNick:String;
		private var _userURL:String;
		private var _userURL_rss:String;
		
		
		/**
		 * Construct a new Question instance.
		 *
		 * @langversion ActionScript 3.0
		 * @playerversion Flash 9
		 * @author Hepp Maccoy 01/26/07
		 * @tiptext
		 */
		public function User()
		{
			// Do Nothing.
		}
		
		
		
		/**
		 * The ID of the User.
		 */
		public function get userId():String
		{
			return _userId;
		}
		
		public function set userId( value:String ):void
		{
			_userId = value;
		}
		
		
		
		/**
		 * The Nick Name of the User.
		 * Note this is not always supplied by the Answsers Api data.
		 */
		public function get userNick():String
		{
			return _userNick;
		}
		
		public function set userNick( value:String ):void
		{
			_userNick = value;
		}
		
		
		
		/**
		 * URL to the User's Avatar.
		 */
		public function get userURL():String
		{
			return _userURL;
		}
		
		public function set userURL( value:String ):void
		{
			_userURL = value;
		}
		
		
		
		/**
		 * URL to the User's RSS Feed
		 */
		public function get userURL_rss():String
		{
			return _userURL_rss;
		}
		
		public function set userURL_rss( value:String ):void
		{
			_userURL_rss = value;
		}
	}
}