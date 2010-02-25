package net.electroland.enteractive.screen {
	
	import flash.display.MovieClip;
	import flash.display.Bitmap;
	import flash.display.Loader;
	import flash.net.URLRequest;
	import flash.events.Event;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import com.yahoo.webapis.weather.*;
	import com.yahoo.webapis.weather.events.*;
	import com.greensock.TweenLite;
	
	/*
	IMAGESWITCHER.as
	by Aaron Siegel, 1-20-2010
	
	Loads the series of images intended to display, and switches
	them according to the current time, sunrise, and sunset.
	
	EnteractiveTimelapse_vs01_00189.png = SUNRISE
	EnteractiveTimelapse_vs01_00263.png = SUNSET
	EnteractiveTimelapse_vs01_00267.png = NIGHT
	
	*/
	
	public class ImageSwitcher extends MovieClip{
		
		private var weatherService:WeatherService;	// used to fetch sunrise/sunset data daily
		private var nightTimer:Timer;				// used to wait from after-sunset to 2am for weather check
		private var morningTimer:Timer;				// used to wait from 2am weather check to just before sunrise
		
		public var sunrise:Date;					// todays sunrise
		public var sunset:Date;						// todays sunset
		public var dayLength:Number;				// seconds in the day between sunrise and sunset
		public var imageOld:Bitmap;					// bottom image
		public var imageNew:Bitmap;					// top image, fading in over time, which becomes bottom image
		
		private var firstImage:Boolean = true;		// ONLY used to indicate the start of the application, so as not to tween in first image
		private var sunriseImage:Number = 189;		// # of sunrise image
		private var sunsetImage:Number = 263;		// # of sunset image
		private var nightImage:Number = 267;		// # of last image
		private var totalDayImages:Number = sunsetImage - sunriseImage;
		private var numTwilightImages:Number = 4;	// # of images used to define twilight period
		private var currentImage:Number;			// # of most recent image displayed
		private var secsPerImage:Number;			// seconds to display each image
		private var imageLoader:Loader;				// loads images one at a time
		private var imagePrefix:String;
		
		public function ImageSwitcher(){
			weatherService = new WeatherService();
			weatherService.addEventListener(WeatherResultEvent.WEATHER_LOADED, weatherLoaded);
			weatherService.addEventListener(WeatherErrorEvent.XML_LOADING, xmlLoading);
			weatherService.addEventListener(WeatherErrorEvent.INVALID_LOCATION, invalidLocation);
			weatherService.getWeather("90020", Units.ENGLISH_UNITS);
			
			imagePrefix = "images/EnteractiveTimelapse_vs01_00";
			imageLoader = new Loader();
			imageLoader.contentLoaderInfo.addEventListener("complete", imageLoaded);
		}
		
		public function weatherLoaded(e:WeatherResultEvent):void{
			sunrise = e.data.current.astronomy.sunrise;
			sunset = e.data.current.astronomy.sunset;
			dayLength = (sunset.time - sunrise.time) / 1000;
			secsPerImage = dayLength / (sunsetImage - sunriseImage);
			//secsPerImage = 2;	// TESTING ONLY
			var now:Date = new Date();												// current time since epoch in milliseconds
			if(now.time > sunrise.time){
				var diff:Number = now.time - sunrise.time;							// time since sunrise in milliseconds
				var numImages:Number = Math.round((diff / 1000) / secsPerImage);	// number of images since sunrise
				currentImage = sunriseImage + numImages;							// is the current image
				if(currentImage > nightImage){										// if it's beyond night image...
					currentImage = nightImage;										// set it to night time
				}
			} else {																// if before sunrise...
				currentImage = nightImage;											// use night time image
			}
			trace("sunrise: " + sunrise);
			trace("sunset:  " + sunset);
			trace("seconds in the day: " + dayLength);
			trace("seconds per image:  " + secsPerImage);
			if(firstImage){
				start();
			} else {
				setMorningTimer()
			}
		}
		
		public function imageLoaded(e:Event):void{
			trace(new Date() + ": image "+currentImage+" loaded");
			imageNew = imageLoader.content as Bitmap;
			if(firstImage){			// first image to be displayed on application start
				var timer:Timer = new Timer(secsPerImage*1000, 1);
				timer.addEventListener("timer", nextImage);
				timer.start();
				firstImage = false;
			} else if(currentImage == sunriseImage){
				imageNew.alpha = 0;
				TweenLite.to(imageNew, secsPerImage*numTwilightImages, {alpha:1, onComplete:imageFadedIn});
			} else {
				imageNew.alpha = 0;
				TweenLite.to(imageNew, secsPerImage, {alpha:1, onComplete:imageFadedIn});
			}
			addChild(imageNew);
		}
		
		public function imageFadedIn():void{
			trace("image "+currentImage+" faded in");
			if(imageOld != null && this.contains(imageOld)){
				removeChild(imageOld);
			}
			imageOld = imageNew;
			addChild(imageOld);
			currentImage++;
			if(currentImage > nightImage){
				setNightTimer();
			} else {
				imageLoader.load(new URLRequest(imagePrefix+String(currentImage)+".png"));
			}
		}
		
		public function xmlLoading(e:WeatherErrorEvent):void{
			trace("xmlLoading error");
		}
		
		public function invalidLocation(e:WeatherErrorEvent):void{
			trace("invalidLocation error");
		}
		
		public function nextImage(e:TimerEvent):void{
			imageFadedIn();
		}
		
		public function setNightTimer():void{
			// when the last image is hit, set a timer for 2am the next day.
			var checkingHour:Number = 2;	// 2am, when the weather checker will run
			var now:Date = new Date();		// current date/time
			trace(now + ": NIGHT TIME");
			var hourdiff:Number = (23 + checkingHour) - now.hours;	// hours until checking hour
			var mindiff:Number = 59 - now.minutes;	// minutes until checking hour
			var secdiff:Number = 59 - now.seconds;	// seconds until checking hour
			var millisTillWeatherCheck:Number = ((hourdiff*60*60*1000) + (mindiff*60*1000) + (secdiff*1000));	// total millis till checking hour
			trace(millisTillWeatherCheck +" milliseconds until weather check");
			nightTimer = new Timer(millisTillWeatherCheck, 1);
			nightTimer.addEventListener("timer", nightTimerEvent);
			nightTimer.start();
		}
		
		public function nightTimerEvent(e:TimerEvent):void{
			// when the 2am timer is triggered, check the weather for new sunrise/sunset data.	
			weatherService.getWeather("90020", Units.ENGLISH_UNITS);
		}
		
		public function setMorningTimer():void{
			// when new sunrise/sunset data comes in, set new timer for sunrise - secsPerDay.
			var now:Date = new Date();		// current date/time
			var millisTillMorning:Number = (sunrise.time - (numTwilightImages*secsPerImage*1000)) - now.time;	// total millis till starting tween into sunrise
			trace(now +": "+ millisTillMorning +" milliseconds until morning twilight tween begins");
			morningTimer = new Timer(millisTillMorning, 1);
			morningTimer.addEventListener("timer", morningTimerEvent);
			morningTimer.start();
		}
		
		public function morningTimerEvent(e:TimerEvent):void{
			currentImage = sunriseImage;
			imageLoader.load(new URLRequest(imagePrefix+String(currentImage)+".png"));
		}
		
		public function start():void{
			imageLoader.load(new URLRequest(imagePrefix+String(currentImage)+".png"));
		}
		
	}
	
}