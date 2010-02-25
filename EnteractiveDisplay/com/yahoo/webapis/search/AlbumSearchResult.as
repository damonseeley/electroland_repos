/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	[Bindable]
	/**
	 * AlbumSearchResult is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 * 
	 */
	public class AlbumSearchResult extends SearchResult
	{
		
		private var _artist:Artist;
		private var _publisher:String;
		
		private var _releaseDate:String;
		
		private var _thumbnail:Thumbnail;
		
		private var _relatedAlbums:Array;
		
	
		
		
		/**
		 * Construct a new AlbumSearchResult instance.
		 *
		 */
		public function AlbumSearchResult()
		{
			// Do Nothing.
		}
		
		
		/**
		 * The thumbnail.
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
		 * Contains a list of related albums that fans of the album might like.
		 */
		public function get relatedAlbums():Array
		{
			return _relatedAlbums;
		}
		
		public function set relatedAlbums( value:Array ):void
		{
			_relatedAlbums = value;
		}
		
		
		
		/**
		 * The artist of the Album.
		 */
		public function get artist():Artist
		{
			return _artist;
		}
		
		public function set artist( value:Artist ):void
		{
			_artist = value;
		} 
		
		/**
		 * The release date of the Album.
		 */
		public function get releaseDate():String
		{
			return _releaseDate;
		}
		
		public function set releaseDate( value:String ):void
		{
			_releaseDate = value;
		} 
		
		/**
		 * The publisher of the content.
		 */
		public function get publisher():String
		{
			return _publisher;
		}
		
		public function set publisher( value:String ):void
		{
			_publisher = value;
		}
		
		
	}
}