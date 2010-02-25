/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers
{
	import com.yahoo.webapis.answers.events.*;
	import com.yahoo.webapis.answers.params.*;
	
	import flash.events.Event;
	import flash.events.EventDispatcher;
	import flash.events.IOErrorEvent;
	import flash.net.URLLoader;
	import flash.net.URLRequest;
	
	//--------------------------------------
	//  Events
	//--------------------------------------
	
	[Event(name="searchByQueryResult",type="com.yahoo.webapis.answers.events.AnswersResultEvent")]
	[Event(name="searchByCategoryResult",type="com.yahoo.webapis.answers.events.AnswersResultEvent")]
	[Event(name="searchByUserResult",type="com.yahoo.webapis.answers.events.AnswersResultEvent")]
	[Event(name="getQuestionResult",type="com.yahoo.webapis.answers.events.AnswersResultEvent")]
	[Event(name="getAnswersResult",type="com.yahoo.webapis.answers.events.AnswersResultEvent")]
	[Event(name="getCommentsResult",type="com.yahoo.webapis.answers.events.AnswersResultEvent")]
	[Event(name="errorEvent",type="com.yahoo.webapis.answers.events.AnswersErrorEvent")]
	
	/**
	 * ActionScript 3 interface to the Yahoo! Answers API.
	 * 
	 * @author Hepp Maccoy, Josh Tynjala
	 */
	public class AnswersService extends EventDispatcher
	{
		
	//--------------------------------------
	//  Static Properties
	//--------------------------------------
	
		/**
		 * @private
		 * The URL used to access the Yahoo! Answers API.
		 */
		private static const API_URL:String = "http://answers.yahooapis.com/AnswersService/V1/";
		
		/**
		 * @private
		 * The base URL used for user profiles.
		 */
		private static const PROFILE_BASE_URL:String = "http://answers.yahoo.com/my/profile?show=";
		
		/**
		 * @private
		 * The base URL used for user RSS feeds.
		 */
		private static const RSS_BASE_URL:String = "http://answers.yahoo.com/rss/userq?kid=";
		
		/**
		 * @private
		 * The namespace used in the XML provided by the answers API.
		 */
 		private static const ANSWERS_NAMESPACE:Namespace = new Namespace("urn:yahoo:answers");
 		
	//--------------------------------------
	//  Constructor
	//--------------------------------------
	
		/**
		 * Constructor.
		 */
 		public function AnswersService(applicationID:String = null)
 		{
 			super();
 			this.applicationID = applicationID;
 		}
		
	//--------------------------------------
	//  Properties
	//--------------------------------------
	
		/**
		 * @private
		 * Storage for the applicationID property.
		 */
 		private var _applicationID:String;
 		
 		/**
 		 * Your Yahoo! Web Services application ID. Required.
 		 * 
 		 * @see http://developer.yahoo.com/wsregapp/
 		 */
 		public function get applicationID():String
 		{
 			return this._applicationID;
 		}
 		
 		/**
 		 * @private
 		 */
 		public function set applicationID(value:String):void
 		{
 			this._applicationID = value;
 		}
		
	//--------------------------------------
	//  Public Methods
	//--------------------------------------
		
		/**
		 * Search questions based on a search term.
		 *
		 * @param query		The search term
		 * @param params	Optional parameters to filter searches
		 */
		public function searchQuestionsByTerm(query:String, params:SearchParams = null):void
		{
			this.handleQueryLoading([], "questionSearch", ("&query=" + query), AnswersResultEvent.SEARCH_BY_QUERY_RESULT, params);
		}
		
		/**
		 * Search questions by category.
		 *
		 * @param categoryID		The ID of the category.
		 * @param params			Optional paramaters to filter searches
		 */
		public function searchQuestionsByCategory(categoryID:String, params:CategoryParams = null):void
		{
			this.handleQueryLoading([], "getByCategory", ("&category_id=" + categoryID), AnswersResultEvent.SEARCH_BY_CATEGORY_RESULT, params);
		}
		
		/**
		 * Search questions by user.
		 *
		 * @param userID		The ID of the user
		 * @param params		Optional parameters to filter searches
		 */
		public function searchQuestionsByUser(userID:String, params:UserParams = null):void
		{
			this.handleQueryLoading([], "getByUser", ("&user_id=" + userID), AnswersResultEvent.SEARCH_BY_USER_RESULT, params);
		}
		
		/**
		 * Return question by question ID.
		 *
		 * @param questionID	The ID of the question
		 */
		public function searchQuestionByID(questionID:String):void
		{
			this.handleQueryLoading([], "getQuestion", ("&question_id=" + questionID), AnswersResultEvent.GET_QUESTION_RESULT);
		}
		
		/**
		 * Return answers by question ID.
		 * 
		 * <p><strong>Important:</strong> Not all questions have answers, before calling this, check the <code>numAnswers</code> property to see if answers are present.</p>
		 *
		 * @param questionID	The ID of the question
		 */
		public function searchAnswers(questionID:String):void
		{
			this.handleQueryLoading(["Answers", 0, 16], "getQuestion", ("&question_id=" + questionID), AnswersResultEvent.GET_ANSWERS_RESULT);
		}
		
		/**
		 * Return comments by question ID.
		 * 
		 * <p><strong>Important:</strong> Not all questions have comments, before calling this, check the <code>numComments</code> property to see if comments are present.
		 *
		 * @param questionID	The ID of the Question.
		 */
		public function searchComments(questionID:String):void
		{
			this.handleQueryLoading(["Comments", 0, 17], "getQuestion", ("&question_id=" + questionID), AnswersResultEvent.GET_COMMENTS_RESULT);
		}
		
		/**
		 * Get the URL of a user's profile.
		 *
		 * @param userID 	The user's ID
		 */
		public function getUserURL(userID:String):String
		{
			if(userID)
			{
				var query:String = PROFILE_BASE_URL + escape(userID);
				return query;
			}
			return "";
		}
		
		/**
		 * Get the URL of a user's RSS Feed.
		 *
		 * @param userID 	The User's ID
		 */
		public function getUserRSS(userID:String):String
		{
			if(userID)
			{
				var query:String = RSS_BASE_URL + escape(userID);
				return query;
			}
			return "";
		}
		
	//--------------------------------------
	//  Private Methods
	//--------------------------------------
		
		/**
		 * @private
		 * Internal handling and loading a Answers API Call.
		 *
		 * @param depth 		An Array containing: 0 = Answers or Comments, 1 = XML Child Depth 1, 2 = XML Child Depth 2.
		 * @param sendMethod 	The String description of the method.
		 * @param query 		The full search query info of the load.
		 * @param dispatchType	The Event dispatch type.
		 * @param params 		Optional value for passing parameterst.
		 */
		private function handleQueryLoading(depth:Array, sendMethod:String, query:String, dispatchType:String, params:Object = null):void
		{
			var sendQuery:String = (API_URL + sendMethod + "?appid=" + this.applicationID + query);
			if(params != null)
			{
				var paramString:String = params.collect();
				sendQuery += ("&" + paramString);
			}
			
			var queryLoader:URLLoader = new URLLoader(new URLRequest(sendQuery));
			queryLoader.addEventListener(Event.COMPLETE, queryLoaderCompleteHandler);
			queryLoader.addEventListener(IOErrorEvent.IO_ERROR, queryLoaderIOErrorHandler);
			
			// XML loading result method
			function queryLoaderCompleteHandler(evtObj:Object):void
			{
				var queryXML:XML = new XML(queryLoader.data);
				
				if(queryXML.ANSWERS_NAMESPACE::Error == undefined)
				{
					// Determine if it's a (Answer or Comment List) or a Question List
					if(depth.length > 0)
					{
						var resultsList:Array = new Array();
						
						// Determine if it's a Answer or Comment
						if(depth[0] == "Answers")
						{
							queryXML.child(depth[1]).child(depth[2]).@type = queryXML.child(0).@type;
							
							//If the Question is Answered, pass along additional data about the ChosenAnswer.
							if(queryXML.child(0).@type == "Answered")
							{
								queryXML.child(depth[1]).child(depth[2]).@ChosenAnswererId = queryXML.child(0).ANSWERS_NAMESPACE::ChosenAnswererId;
								queryXML.child(depth[1]).child(depth[2]).@ChosenAnswerTimestamp = queryXML.child(0).ANSWERS_NAMESPACE::ChosenAnswerTimestamp;
							}
							
							resultsList = returnAnswersFromXML(queryXML.child(depth[1]).child(depth[2]));
						}
						else
						{
							resultsList = returnCommentsFromXML(queryXML.child(depth[1]).child(depth[2]));
						}
						
						if(resultsList.length > 0)
						{
							var resultsEvent:AnswersResultEvent = new AnswersResultEvent(dispatchType, resultsList);
							dispatchEvent(resultsEvent);
						}
						else
						{
							if(depth[0] == "Answers")
							{
								dispatchError("No Answers exist or this is not a valid Question ID. Always check if a Question has answers first with 'Question.numAnswers'.", "NONEXISTANT");
							}
							else
							{
								dispatchError("No Comments exist or this is not a valid Question ID. Always check if a Question has comments first with 'Question.numComments'.", "NONEXISTANT");
							}
						}
					}
					else
					{
						// Format and Dispatch the Normal Question Objects
						queryXML = new XML(queryLoader.data);
						formatAndDispatch(queryXML, dispatchType);
					}
				}
				else
				{
					dispatchError("Api Returned Error Message: " + queryXML.ANSWERS_NAMESPACE::Message, "API_RESPONSE");
				}
			    
			}
			
			function queryLoaderIOErrorHandler(event:IOErrorEvent):void
			{
			    dispatchError("Error Loading: " + sendQuery, "XML_LOADING");
			}
		}
		
		/**
		 * @private
		 * Recurse results and transcribe into Answers Objects, then dispatch update Event.
		 *
		 * @param originalXML		The XML to transcribe into Objects.
		 * @param dispatchType		The term of the dispatched Event.
		 */
		private function formatAndDispatch(originalXML:XML, dispatchType:String):void
		{
			var resultsList:Array = new Array();
		    
		    for each (var questionNode:XML in originalXML.children())
			{
				var question:Question = new Question();
				
				if(questionNode.@id != undefined && questionNode.@id != ""){ question.success = true; };
				
				// **Regular Question Properties
				question.id = questionNode.@id;
				question.subject = questionNode.ANSWERS_NAMESPACE::Subject;
				question.content = questionNode.ANSWERS_NAMESPACE::Content;
				question.link = questionNode.ANSWERS_NAMESPACE::Link;
				question.authorPhotoURL = questionNode.ANSWERS_NAMESPACE::UserPhotoURL;
				
				//convert to date
				question.date = new Date(String(questionNode.ANSWERS_NAMESPACE::Date).replace(/-/g, "/"));
				question.timestamp = new Date(String(questionNode.ANSWERS_NAMESPACE::Timestamp).replace(/-/g, "/"));
					
				question.numAnswers = questionNode.ANSWERS_NAMESPACE::NumAnswers;
				question.numComments = questionNode.ANSWERS_NAMESPACE::NumComments;
				question.type = questionNode.@type;
				
				// **Object Properties
				// --Category Object
				var questionCategory:Category = new Category();
				questionCategory.id = questionNode.ANSWERS_NAMESPACE::Category.@id;
				questionCategory.name = questionNode.ANSWERS_NAMESPACE::Category;
				// Add the new Category Object to our Question Object.
				question.inCategory = questionCategory;
				
				// --User Object for the Questions Author
				var questionUser:User = new User();
				questionUser.userId = questionNode.ANSWERS_NAMESPACE::UserId;
				questionUser.userNick = questionNode.ANSWERS_NAMESPACE::UserNick;
				questionUser.userURL = getUserURL(questionUser.userId);
				questionUser.userURL_rss = getUserRSS(questionUser.userId);
				// Add the new User Object to our Question Object.
				question.author = questionUser;
				
				// --User Object for the ChosenAnswers Author
				var answerUser:User = new User();
				answerUser.userId = questionNode.ANSWERS_NAMESPACE::ChosenAnswererId;
				answerUser.userNick = questionNode.ANSWERS_NAMESPACE::ChosenAnswererNick; 
				answerUser.userURL = getUserURL(answerUser.userId);
				answerUser.userURL_rss = getUserRSS(answerUser.userId);
				
				// Only pass a ChosenAnswer object when the Question is actually answered.
				if(question.type == "Answered")
				{
					question.hasChosenAnswer = true;
					
					// --Chosen Answer Object
					var chosenAnswer:ChosenAnswer = new ChosenAnswer();
					chosenAnswer.content = questionNode.ANSWERS_NAMESPACE::ChosenAnswer;
					chosenAnswer.author = answerUser; // !!! Perhaps clone Object, not reference.
					chosenAnswer.postedTimestamp = questionNode.ANSWERS_NAMESPACE::ChosenAnswerTimestamp;
					chosenAnswer.awardedTimestamp = questionNode.ANSWERS_NAMESPACE::ChosenAnswerAwardTimestamp;
					// Add the new Answer Object to our Question Object.
					question.chosenAnswer = chosenAnswer;
				}
				else
				{
					question.hasChosenAnswer = false;
					
					// Set the ChosenAnswer to null if it doesn't exist
					question.chosenAnswer = null;
				}
				
				
				if(questionNode.ANSWERS_NAMESPACE::Answers != undefined)
				{
					var answerList:Array = returnAnswersFromXML(questionNode.ANSWERS_NAMESPACE::Answers);
					question.hasAnswers = true;
					question.answers = answerList;
				}
				else
				{
					question.hasAnswers = false;
					question.answers = null;
				}
				
				// Add the Question Object to our Data Array.
				resultsList.push(question);
			}
			
			if(resultsList.length < 1 && resultsList.success != undefined)
			{
				dispatchError("No Results Found", "NO_RESULTS");
			}
			
			var resultsEvent:AnswersResultEvent = new AnswersResultEvent(dispatchType, resultsList);
			this.dispatchEvent(resultsEvent);
		}
		
		
		/**
		 * @private
		 * Internal method for dispatching Error Events.
		 *
		 * @param message 			The Error Message
		 * @param type 				The Event Type
		 */
		private function dispatchError(message:String, type:String):void
		{
			var aResults:AnswersErrorEvent = new AnswersErrorEvent(AnswersErrorEvent.ERROR_EVENT, message, type);
			this.dispatchEvent(aResults);
		}
		
		
		/**
		 * @private
		 * Internal method for returning Answers objects from a Answers XMLList.
		 *
		 * @param originalXML	The XML List containing Answer nodes.
		 */
		private function returnAnswersFromXML(originalXML:XMLList):Array
		{
			// Create the Answer Objects
			var resultsList:Array = new Array();
			for each(var answersNode:XML in originalXML.children())
			{
				var answer:Answer = new Answer();
				
				answer.best = answersNode.ANSWERS_NAMESPACE::Best;
				answer.content = answersNode.ANSWERS_NAMESPACE::Content;
				answer.date = new Date(String(answersNode.ANSWERS_NAMESPACE::Date).replace(/-/g, "/"));
				answer.reference = answersNode.ANSWERS_NAMESPACE::Reference;
				answer.timestamp = answersNode.ANSWERS_NAMESPACE::Timestamp;
				
				// --User Object
				var user:User = new User();
				user.userId = answersNode.ANSWERS_NAMESPACE::UserId;
				user.userNick = answersNode.ANSWERS_NAMESPACE::UserNick;
				user.userURL = getUserURL(user.userId);
				user.userURL_rss = getUserRSS(user.userId);
				answer.author = user;
				
				// If Answer is the 'ChosenAnswer'.
				answer.isChosenAnswer = false;
				if((String(originalXML.@type) == "Answered") && (user.userId == String(originalXML.@ChosenAnswererId)) && (answer.timestamp == originalXML.@ChosenAnswerTimestamp))
				{
					answer.isChosenAnswer = true;
				}
				
				resultsList.push(answer);
			}
			return resultsList;
		}
		
		/**
		 * @private
		 * Internal method for returning Comment objects from a Comments XMLList.
		 *
		 * @param originalXML		The XMLList containing comment nodes.
		 */
		private function returnCommentsFromXML(originalXML:XMLList):Array
		{
			// Create the Comment Objects
			var resultsList:Array = new Array();
			for each(var commentNode:XML in originalXML.children())
			{
				var comment:Comment = new Comment();
				
				comment.content = commentNode.ANSWERS_NAMESPACE::Content;
				comment.date = commentNode.ANSWERS_NAMESPACE::Date;
				comment.timestamp = commentNode.ANSWERS_NAMESPACE::Timestamp;
				
				// --User Object
				var answerUser:User = new User();
				answerUser.userId = commentNode.ANSWERS_NAMESPACE::UserId;
				answerUser.userNick = commentNode.ANSWERS_NAMESPACE::UserNick;
				answerUser.userURL = getUserURL(answerUser.userId);
				answerUser.userURL_rss = getUserRSS(answerUser.userId);
				comment.author = answerUser;
				
				resultsList.push(comment);
			}
			return resultsList;
		}
	}
}