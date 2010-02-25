/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers
{
	import com.yahoo.webapis.answers.Category;
	import com.yahoo.webapis.answers.ChosenAnswer;
	import com.yahoo.webapis.answers.Answer;
	import com.yahoo.webapis.answers.User;
	
	/**
	 * Question is a ValueObject for the Answers API.
	 */
	public class Question
	{
		public var success:Boolean = false;
		private var _id:String;
		private var _subject:String;
		private var _content:String;
		private var _author:User;
		private var _authorPhotoURL:String;
		private var _link:String;
		private var _date:Date;
		private var _timestamp:Date;
		private var _inCategory:Category;
		private var _numAnswers:Number;
		private var _numComments:Number;
		private var _type:String;
		private var _hasChosenAnswer:Boolean;
		private var _chosenAnswer:ChosenAnswer;
		
		// Only Questions created by a SingleQuestion search recieve:
		private var _hasAnswers:Boolean;
		private var _answers:Array;
		
		
		/**
		 * Construct a new Question instance.
		 *
		 * @langversion ActionScript 3.0
		 * @playerversion Flash 9
		 * @author Hepp Maccoy 01/04/07
		 * @tiptext
		 */
		public function Question()
		{
			// Do Nothing.
		}
		
		
		
		/**
		 * The literal Question.
		 */
		public function get subject():String
		{
			return _subject;
		}
		
		public function set subject( value:String ):void
		{
			_subject = value;
		}
		
		
		
		/**
		 * The Content of the Question.
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
		 * The User who posted the Question.
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
		 * The URL to the author's avatar photo.
		 */
		public function get authorPhotoURL():String
		{
			return _authorPhotoURL;
		}
		
		public function set authorPhotoURL( value:String ):void
		{
			_authorPhotoURL = value;
		}
		
		
		/**
		 * The url link to the Question.
		 */
		public function get link():String
		{
			return _link;
		}
		
		public function set link( value:String ):void
		{
			_link = value;
		}
		
		
		
		/**
		 * The Date of the Question.
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
		 * The Timestamp of the Question.
		 */
		public function get timestamp():Date
		{
			return _timestamp;
		}
		
		public function set timestamp( value:Date ):void
		{
			_timestamp = value;
		}
		
		
		
		/**
		 * The Category the Question is in.
		 */
		public function get inCategory():Category
		{
			return _inCategory;
		}
		
		public function set inCategory( value:Category ):void
		{
			_inCategory = value;
		}
		
		
		/**
		 * The amount of Answers on the Question.
		 */
		public function get numAnswers():Number
		{
			return _numAnswers;
		}
		
		public function set numAnswers( value:Number ):void
		{
			_numAnswers = value;
		}
		
		
		
		/**
		 * The amount of Comments on the Question.
		 */
		public function get numComments():Number
		{
			return _numComments;
		}
		
		public function set numComments( value:Number ):void
		{
			_numComments = value;
		}
		
		
		
		/**
		 * The Type of the Question, either 'Answered', 'Voting', or 'Open'
		 */
		public function get type():String
		{
			return _type;
		}
		
		public function set type( value:String ):void
		{
			_type = value;
		}
		
		
		
		
		/**
		 * An easy flag boolean to determine if there has been a chosenAnswer yet.
		 */
		public function get hasChosenAnswer():Boolean
		{
			return _hasChosenAnswer;
		}
		
		public function set hasChosenAnswer( value:Boolean ):void
		{
			_hasChosenAnswer = value;
		}
		
		
		
		/**
		 * The id of the Question.
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
		 * The Chosen Answer of the Question.
		 */
		public function get chosenAnswer():ChosenAnswer
		{
			return _chosenAnswer;
		}
		
		public function set chosenAnswer( value:ChosenAnswer ):void
		{
			_chosenAnswer = value;
		}
		
		
		
		
		/**
		 * The Chosen Answer of the Question.
		 */
		public function get hasAnswers():Boolean
		{
			return _hasAnswers;
		}
		
		public function set hasAnswers( value:Boolean ):void
		{
			_hasAnswers = value;
		}
		
		
		
		/**
		 * The Chosen Answer of the Question.
		 */
		public function get answers():Array
		{
			return _answers;
		}
		
		public function set answers( value:Array ):void
		{
			_answers = value;
		}
	}
}