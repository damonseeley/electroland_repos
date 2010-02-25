/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	/**
	 * Song is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
	public class Song
	{
		
		private var _name:String;
		private var _id:String;
		
		private var _album:Album;
		private var _artist:Artist;
		private var _publisher:String;
		private var _duration:Number;
		private var _releaseDate:String;
		private var _index:uint;
		private var _thumbnail:Thumbnail;
		/**
		 * Construct a new Song instance.
		 */
		public function Song()
		{
			// Do Nothing.
		}
		
		/**
		 * The name of the Song.
		 */
		public function get name():String
		{
			return _name;
		}
		
		public function set name( value:String ):void
		{
			_name = value;
		} 
		
		/**
		 * The id of the Song.
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
		 * The album of the Song.
		 */
		public function get album():Album
		{
			return _album;
		}
		
		public function set album( value:Album ):void
		{
			_album = value;
		} 
		
		/**
		 * The artist of the Song.
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
		 * The index of the Song in the album (track number).
		 */
		public function get index():uint
		{
			return _index;
		}
		
		public function set index( value:uint ):void
		{
			_index = value;
		} 
		
		/**
		 * The release date of the Song.
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
		
		/**
		 * The duration in seconds.
		 */
		public function get duration():Number
		{
			return _duration;
		}
		
		public function set duration( value:Number ):void
		{
			_duration = value;
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
		 * The label of the Song.
		 */
		public function toString():String
		{
			return _name;
		}
	}
}