/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers
{
	/**
	 * ChosenAnswer is a ValueObject for the Answers API.
	 */
	public class ChosenAnswer
	{
		private var _content:String;
		private var _author:User;
		private var _postedTimestamp:Number;
		private var _awardedTimestamp:Number;
		
		
		/**
		 * Construct a new ChosenAnswer instance. A ChosenAnswer is like a Answers object, but it contains less details and is a reference in addition to a Answer Object when supplied for that Question.
		 * A ChosenAnswer is always returned with a Question if it is resolved.
		 *
		 * @langversion ActionScript 3.0
		 * @playerversion Flash 9
		 * @author Hepp Maccoy 01/04/07
		 * @tiptext
		 */
		public function ChosenAnswer()
		{
			// Do Nothing.
		}
		
		
		
		/**
		 * The content of the Answer.
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
		 * The User of the Answer.
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
		 * The Posted Date of the Answer.
		 */
		public function get postedTimestamp():Number
		{
			return _postedTimestamp;
		}
		
		public function set postedTimestamp( value:Number ):void
		{
			_postedTimestamp = value;
		}
		
		
		
		/**
		 * The Awarded Date of the Answer.
		 */
		public function get awardedTimestamp():Number
		{
			return _awardedTimestamp;
		}
		
		public function set awardedTimestamp( value:Number ):void
		{
			_awardedTimestamp = value;
		}
	}
}