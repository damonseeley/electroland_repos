package net.electroland.kioskengine {
	import flash.utils.Timer;
	import flash.events.*;
	
	public class EffectTimer extends EventDispatcher{
		private var effectLanguage:String;
		private var effectGender:String;
		private var effectAge:String;
		private var effectType:String;
		private var effectDuration:int;
		private var timerDelay:int;
		private var timedEvent:Boolean;
		private var onExit:Boolean;
		private var onComplete:Boolean;
		private var onVideoEnd:Boolean;
		private var timer:Timer;
		private var targetX:int;
		private var targetY:int;
		
		/*
		
		EFFECTTIMER.as
		by Aaron Siegel, 7-3-09
		
		EffectTimer is necessary for retaining the properties for effects
		(such as duration, width/height targets, position targets, etc.)
		to be return as EffectEvents, which are ultimately executed as
		tweens within the Item superclass.
		
		*/
				
		public function EffectTimer(itemEffectElement:XML){
			effectLanguage = itemEffectElement.attribute("Language");
			effectGender = itemEffectElement.attribute("Gender");
			effectAge = itemEffectElement.attribute("Age");
			effectType = itemEffectElement.attribute("Action");
			effectDuration = Number(itemEffectElement.attribute("Duration"));
			if(effectType == "movein" || effectType == "moveout"){
				targetX = itemEffectElement.attribute("x");
				targetY = itemEffectElement.attribute("y");
			}
			if(itemEffectElement.attribute("Time") == "onExit"){
				timedEvent = false;		// must trigger effect when unit is told to exit
				onExit = true;
			} else if(itemEffectElement.attribute("Time") == "onComplete"){	// used with buttonpairlists
				timedEvent = false;		// must trigger effect when unit is told buttonpairlist has been completed
				onComplete = true;
			} else if(itemEffectElement.attribute("Time") == "videoEnd"){
				timedEvent = false;		// must trigger when video completes
				onVideoEnd = true;
			} else {
				timedEvent = true;
				var timesplit:Array = itemEffectElement.attribute("Time").split(":");
				var hourms:int = Number(timesplit[0]) * 60 * 60 * 1000;
				var minsms:int = Number(timesplit[1]) * 60 * 1000;
				var secondsms:int = Number(timesplit[2]) * 1000;
				timerDelay = hourms + minsms + secondsms + Number(timesplit[3]);
				timer = new Timer(timerDelay, 1);
				timer.addEventListener("timer", timerFinished);
			}
		}
		
		public function getDelay():int{
			return timerDelay;
		}
		
		public function getDuration():int{
			return effectDuration;
		}
		
		public function getLanguage():String{
			return effectLanguage;
		}
		
		public function getType():String{
			return effectType;
		}
		
		public function isTimedEvent():Boolean{
			return timedEvent;
		}
		
		public function isOnExit():Boolean{
			return onExit;
		}
		
		public function isOnComplete():Boolean{
			return onComplete;
		}
		
		public function isOnVideoEnd():Boolean{
			return onVideoEnd;
		}
		
		public function resetTimer():void{
			if(timer != null){
				timer.reset();
			}
		}
		
		public function startEffect():void{
			//trace("timer finished!");
			var attributes:Object = new Object();
			attributes["language"] = effectLanguage;
			attributes["gender"] = effectGender;
			attributes["age"] = effectAge;
			attributes["type"] = effectType;
			attributes["duration"] = effectDuration
			attributes["x"] = targetX;
			attributes["y"] = targetY;
			dispatchEvent(new EffectEvent(effectType, attributes));
		}
		
		public function startTimer():void{
			if(timer != null){
				timer.start();
			}
		}
		
		public function timerFinished(event:TimerEvent):void{
			startEffect();
		}
		
	}
	
}