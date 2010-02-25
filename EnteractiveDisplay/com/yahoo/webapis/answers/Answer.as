/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers
{
	import com.yahoo.webapis.answers.User;
	
	/**
	 * Answer is a ValueObject for the Answers API.
	 */
	public class Answer
	{
		private var _content:String;
		private var _timestamp:Number;
		private var _reference:String;
		private var _best:String;
		private var _author:User;
		private var _date:Date;
		private var _isChosenAnswer:Boolean;
		
		
		/**
		 * Construct a new Question instance.
		 *
		 * @langversion ActionScript 3.0
		 * @playerversion Flash 9
		 * @author Hepp Maccoy 01/04/07
		 * @tiptext
		 */
		public function Answer()
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
		 * The Posted timestamp of the Answer.
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
		 * An optional reference link.
		 */
		public function get reference():String
		{
			return _reference;
		}
		
		public function set reference( value:String ):void
		{
			_reference = value;
		}
		
		
		
		
		/**
		 * The Rating 'Best' for the Answer.
		 */
		public function get best():String
		{
			return _best;
		}
		
		public function set best( value:String ):void
		{
			_best = value;
		}
		
		
		
		
		/**
		 * The User object of the author of the Answer.
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
		 * The Awarded Date of the Answer.
		 */
		public function get date():Date
		{
			return _date;
		}
		
		public function set date( value:Date ):void
		{
			_date = value;
		}
		
		
		/**
		 * If the Answer has been picked as the 'chosen answer' for the Question's author.
		 */
		public function get isChosenAnswer():Boolean
		{
			return _isChosenAnswer;
		}
		
		public function set isChosenAnswer( value:Boolean ):void
		{
			_isChosenAnswer = value;
		}
	}
}