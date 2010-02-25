/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	/**
	 * Rating is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
	 
	public class Rating
	{
		private var _numReviews:uint;
		private var _numRatings:uint;
		private var _averageRating:Number;
		private var _lastReviewDate:Date;
		private var _summary:String;
		
		/**
		 * Construct a new Rating instance.
		 */
		public function Rating()
		{
			// Do Nothing.
		}
		

		/**
		 * The total number of reviews.
		 */
		public function get numReviews():uint
		{
			return _numReviews;
		}
		
		public function set numReviews( value:uint ):void
		{
			_numReviews = value;
		}
		
		/**
		 * The total number of ratings.
		 */
		public function get numRatings():uint
		{
			return _numRatings;
		}
		
		public function set numRatings( value:uint ):void
		{
			_numRatings = value;
		}
		
		/**
		 * The average rating.
		 */
		public function get averageRating():Number
		{
			return _averageRating;
		}
		
		public function set averageRating( value:Number ):void
		{
			_averageRating = value;
		}
		
		
		/**
		 * The Date of the last review.
		 */
		public function get lastReviewDate():Date
		{
			return _lastReviewDate;
		}
		
		public function set lastReviewDate( value:Date ):void
		{
			_lastReviewDate = value;
		}
		
		/**
		 * The summary of the last review.
		 */
		public function get summary():String
		{
			return _summary;
		}
		
		public function set summary( value:String ):void
		{
			_summary = value;
		}
		
		public function toString():String
		{
			return summary;
		}
		
	}
}