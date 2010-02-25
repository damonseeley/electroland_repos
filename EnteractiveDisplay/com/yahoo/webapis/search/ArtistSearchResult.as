/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	[Bindable]
	/**
	 * ArtistSearchResult is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 * 
	 */
	public class ArtistSearchResult extends SearchResult
	{
		
		private var _url:String;
		
		
		private var _thumbnail:Thumbnail;
		
		private var _relatedArtists:Array;
		
		private var _popularSongs:Array;
		
		
		
		/**
		 * Construct a new ArtistSearchResult instance.
		 *
		 */
		public function ArtistSearchResult()
		{
			// Do Nothing.
		}
		
		
		
		/**
		 * The thumbnail of the ArtistSearchResult.
		 */
		public function get thumbnail():Thumbnail
		{
			return _thumbnail;
		}
		
		public function set thumbnail( value:Thumbnail ):void
		{
			_thumbnail = value;
		}
		
		
		/**
		 * The  url to the YahooMusicPage.
		 */
		public function get url():String
		{
			return _url;
		}
		
		public function set url( value:String ):void
		{
			_url = value;
		}
		
		/**
		 * The click url to the YahooMusicPage.
		 */
		public function get clickURL():String
		{
			return _url;
		}
		
		public function set clickURL( value:String ):void
		{
			_url = value;
		}
		
		/**
		 * Contains a list of related artists that fans of the artist in this Result might like.
		 */
		public function get relatedArtists():Array
		{
			return _relatedArtists;
		}
		
		public function set relatedArtists( value:Array ):void
		{
			_relatedArtists = value;
		}
		
		
		/**
		 * Contains a list of some popular songs by this artist.
		 */
		public function get popularSongs():Array
		{
			return _popularSongs;
		}
		
		public function set popularSongs( value:Array ):void
		{
			_popularSongs = value;
		}
		
		
	}
}