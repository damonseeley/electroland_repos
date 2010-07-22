package net.electroland.kioskengine  {
	import flash.display.Sprite;
	import flash.events.*;
	import fl.transitions.Tween;
	import fl.transitions.easing.*;
	import flash.display.Stage;
	import flash.media.Sound;
	import flash.media.SoundChannel;
	import flash.media.SoundTransform;
    import flash.net.URLRequest;
	import flash.utils.Timer;
	import gs.*;
	
	public class Item extends Sprite {
		//
		protected var conductor:Conductor;
		private var itemID:String;
		private var itemName:String;
		private var itemDescription:String;
		private var itemLayer:int;
		private var itemWidth:int;
		private var itemHeight:int;
		private var itemType:String;
		private var itemHorizontalAlign:String;
		private var itemVerticalAlign:String;
		private var buttonType:String;
		private var singleUse:Boolean;
		//private var alphaTween:Tween;	// fade in/out
		private var alphaTween:TweenLite;
		//private var widthTween:Tween;	// pop in/out
		//private var heightTween:Tween;
		private var sizeTween:TweenLite;
		//private var xPosTween:Tween;	// move in/out
		//private var yPosTween:Tween;
		private var posTween:TweenLite;
		protected var itemEntries:Object = new Object();
		protected var fadeInEvents:Array = new Array();		// multiple entries per language
		protected var fadeOutEvents:Array = new Array();
		protected var popInEvents:Array = new Array();
		protected var popOutEvents:Array = new Array();
		protected var moveInEvents:Array = new Array();
		protected var moveOutEvents:Array = new Array();
		protected var cutInEvents:Array = new Array();
		protected var cutOutEvents:Array = new Array();
		protected var activateEvents:Array = new Array();
		protected var deactivateEvents:Array = new Array();
		protected var interactive:Boolean;
		private var clickable:Boolean;
		private var soundPlayer:Sound 
		protected var externalButton:Boolean;
		protected var pressed:Boolean = false;	// used to prevent double-click
		protected var contentLoadedCallback:Function;
		public var deactivated:Boolean = false;	// kludge solution to external button deactivation
		protected var startAudioCallback:Function;	// passed back from AudioItem
		protected var stopAudioCallback:Function;
		protected var clickDelay:Timer;	// delay to prevent accidental double-click
		
		/*
	
		ITEM.as
		by Aaron Siegel, 7-3-09
		
		Item is intended to act as a superclass for all displayable objects within a Unit.
		Video, Images, Text, and Buttons should all extend Item to maintain uniformity.
		Items refer to the event objects by language, and either trigger them by timer or
		onExit.
		
		*/
		
		public function Item(conductor:Conductor) {
			this.conductor = conductor;
			alpha = 0;
			interactive = true;	// true so that everything can be dragged
			clickable = false;
		}
		
		public function addContentListener(f:Function):void{
			contentLoadedCallback = f;
		}
		
		private function addMouseListener():void{
			if(interactive && !clickable){
				addEventListener(MouseEvent.MOUSE_DOWN, mousePressed);
				addEventListener(MouseEvent.MOUSE_UP, mouseReleased);
				clickable = true;
			}
		}
		
		protected function clickDelayListener(event:TimerEvent):void{
			pressed = false;	// allow button to be pressed again
		}
		
		public function effectEventHandler(event:EffectEvent):void{
			//trace(event.getType());
			if(event.getType() == "fadein"){
				fadeUp(event.getAttributes().duration);
			} else if(event.getType() == "fadeout"){
				fadeDown(event.getAttributes().duration);
			} else if(event.getType() == "movein"){
				moveTo(event.getAttributes().x, event.getAttributes().y, event.getAttributes().duration);
			} else if(event.getType() == "moveout"){
				moveTo(event.getAttributes().x, event.getAttributes().y, event.getAttributes().duration);
			} else if(event.getType() == "popin"){
				popIn(event.getAttributes().duration);
			} else if(event.getType() == "popout"){
				popOut(event.getAttributes().duration);
			} else if(event.getType() == "cutin"){
				cutIn();
			} else if(event.getType() == "cutout"){
				cutOut();
			} 
		}
		
		public function getID():String{
			return itemID;
		}
		
		public function getName():String{
			return itemName;
		}
		
		public function getType():String{
			return itemType;
		}
		
		public function getDescription():String{
			return itemDescription;
		}
		
		public function getLayer():int{
			return itemLayer;
		}
		
		public function getWidth():int{
			return itemWidth;
		}
		
		public function getHeight():int{
			return itemHeight;
		}
		
		public function getHorizontalAlign():String{
			return itemHorizontalAlign;
		}
		
		public function getVerticalAlign():String{
			return itemVerticalAlign;
		}
		
		public function getButtonType():String{
			return buttonType;
		}
		
		public function getItemEntries():Object{
			return itemEntries;
		}
		
		public function isSingleUse():Boolean{
			return singleUse;
		}
		
		public function setID(itemID:String):void{
			this.itemID = itemID;
		}
		
		public function setName(itemName:String):void{
			this.itemName = itemName;
		}
		
		public function setType(itemType:String):void{
			this.itemType = itemType;
		}
		
		public function setDescription(itemDescription:String):void{
			this.itemDescription = itemDescription;
		}
		
		public function setLayer(itemLayer:int):void{
			this.itemLayer = itemLayer;
		}
		
		public function setWidth(itemWidth:int):void{
			this.itemWidth = itemWidth;
		}
		
		public function setHeight(itemHeight:int):void{
			this.itemHeight = itemHeight;
		}
		
		public function cutIn():void{
			if(itemType == "Audio"){
				//if(startAudioCallback != null){
					startAudioCallback();
				//}
			}
			alpha = 1;
			addMouseListener();
		}
		
		public function cutOut():void{
			alpha = 0;
			removeMouseListener();
		}
		
		public function fadeUp(fadeDuration:int):void{
			//alphaTween = new Tween(this, "alpha", Strong.easeOut, 0, 1, fadeDuration/1000, true);
			alphaTween = new TweenLite(this, fadeDuration/1000, {alpha:1}); 
			addMouseListener();
		}
		
		public function fadeDown(fadeDuration:int):void{
			//alphaTween = new Tween(this, "alpha", Strong.easeOut, 1, 0, fadeDuration/1000, true);
			alphaTween = new TweenLite(this, fadeDuration/1000, {alpha:0}); 
		}
		
		public function mousePressed(event:MouseEvent):void{
			//trace("go to "+buttonEntries[conductor.getLanguage()].link);
			if(conductor.isEditMode()){
				// EDIT MODE ALLOWS THE USER TO DRAG THE BUTTON
				if(itemType != "ButtonPair" && itemType != "ButtonPairList"){
					startDrag();
				}
			} else {
				if(itemType == "Button"){
					// PLAY OUT THE ACTION OF THIS BUTTON
					if(!externalButton){
						if(!pressed){
							// play indicator sound
							soundPlayer = new Sound();
							soundPlayer.addEventListener(Event.COMPLETE, soundLoaded);
							soundPlayer.addEventListener(IOErrorEvent.IO_ERROR, soundIOError);
							soundPlayer.load(new URLRequest(conductor.getButtonSound()));
							
							if(itemEntries.hasOwnProperty("all")){
								dispatchEvent(new ItemEvent(itemEntries["all"].action, itemEntries["all"].link));
							} else {
								dispatchEvent(new ItemEvent(itemEntries[conductor.getLanguage()].action, itemEntries[conductor.getLanguage()].link));
							}
							
							conductor.logItemEvent(" ", " ", getName(), getDescription(), "button pressed");
							
							pressed = true;
							clickDelay = new Timer(1500,1);
							clickDelay.addEventListener("timer", clickDelayListener);
							clickDelay.start();
						}
					} else {
						if(!deactivated){
							if(!pressed){
								// play indicator sound
								soundPlayer = new Sound();
								soundPlayer.addEventListener(Event.COMPLETE, soundLoaded);
								soundPlayer.addEventListener(IOErrorEvent.IO_ERROR, soundIOError);
								soundPlayer.load(new URLRequest(conductor.getButtonSound()));
							}
						}
					}
				}
			}
		}
		
		public function mouseReleased(event:MouseEvent):void{
			if(conductor.isEditMode()){
				trace("POSITION DATA: "+ itemName+" x: "+x+" y: "+y);
			}
			stopDrag();
		}
		
		public function moveTo(targetx:int, targety:int, duration:int):void{
			//xPosTween = new Tween(this, "x", Strong.easeOut, x, targetx, duration/1000, true);
			//yPosTween = new Tween(this, "y", Strong.easeOut, y, targety, duration/1000, true);
			posTween = new TweenLite(this, duration/1000, {x:targetx, y:targety});
			addMouseListener();
		}
		
		public function popIn(duration:int):void{
			//widthTween = new Tween(this, "width", Strong.easeOut, 1, itemWidth, duration/1000, true);
			//heightTween = new Tween(this, "height", Strong.easeOut, 1, itemHeight, duration/1000, true);
			sizeTween = new TweenLite(this, duration/1000, {width:itemWidth, height:itemHeight});
			addMouseListener();
		}
		
		public function popOut(duration:int):void{
			//widthTween = new Tween(this, "width", Strong.easeOut, itemWidth, 1, duration/1000, true);
			//heightTween = new Tween(this, "height", Strong.easeOut, itemHeight, 1, duration/1000, true);
			sizeTween = new TweenLite(this, duration/1000, {width:0, height:0});
		}
		
		protected function parseXML(itemElement:XML):void{
			//trace(itemElement);
			itemID = itemElement.attribute("ID");
			itemName = itemElement.attribute("Name");
			itemType = itemElement.attribute("Type");
			itemDescription = itemElement.attribute("Description");
			//itemLayer = Number(itemElement.attribute("Layer"));
			itemHorizontalAlign = itemElement.attribute("HorizontalAlign");
			itemVerticalAlign = itemElement.attribute("VerticalAlign");
			if(itemType == "Button"){
				buttonType = itemElement.attribute("ButtonType");
				if(itemElement.attribute("SingleUse") == "true"){
					singleUse = true;
				} else {
					singleUse = false;
				}
			}
			if(itemElement.attribute("x").indexOf("%", 0) > 0){
				x = Number(itemElement.attribute("x").split("%")[0])/100 * conductor.parent.stage.stageWidth;
			} else {
				x = Number(itemElement.attribute("x"));
			}
			if(itemElement.attribute("y").indexOf("%", 0) > 0){
				y = Number(itemElement.attribute("y").split("%")[0])/100 * conductor.parent.stage.stageHeight;
			} else {
				y = Number(itemElement.attribute("y"));
			}
			if(itemElement.attribute("width").indexOf("%", 0) > 0){
				itemWidth = Number(itemElement.attribute("width").split("%")[0])/100 * conductor.parent.stage.stageWidth;
				//trace(itemWidth);
			} else {
				itemWidth = Number(itemElement.attribute("width"));
			}
			if(itemElement.attribute("height").indexOf("%", 0) > 0){
				itemHeight = Number(itemElement.attribute("height").split("%")[0])/100 * conductor.parent.stage.stageHeight;
			} else {
				itemHeight = Number(itemElement.attribute("height"));
			}
			for each (var itemEventElement:XML in itemElement.ItemEvent){
				//trace(itemEventElement.attribute("Action"));
				parseEffects(itemEventElement);
			}
			for each (var itemEntryElement:XML in itemElement.ItemEntry){
				parseItemEntries(itemEntryElement);
			}
		}
		
		protected function parseEffects(itemEventElement:XML):void{
			var effectTimer:EffectTimer = new EffectTimer(itemEventElement);
			effectTimer.addEventListener(EffectEvent.TIMER_EVENT, effectEventHandler);
			if(itemEventElement.attribute("Action") == "fadein"){
				//fadeInEvents[effectTimer.getLanguage()] = effectTimer;
				fadeInEvents.push(effectTimer);
			} else if(itemEventElement.attribute("Action") == "fadeout"){
				//fadeOutEvents[effectTimer.getLanguage()] = effectTimer;
				fadeOutEvents.push(effectTimer);
			} else if(itemEventElement.attribute("Action") == "movein"){
				//moveInEvents[effectTimer.getLanguage()] = effectTimer;
				moveInEvents.push(effectTimer);
			} else if(itemEventElement.attribute("Action") == "moveout"){
				//moveOutEvents[effectTimer.getLanguage()] = effectTimer;
				moveOutEvents.push(effectTimer);
			} else if(itemEventElement.attribute("Action") == "popin"){
				//popInEvents[effectTimer.getLanguage()] = effectTimer;
				popInEvents.push(effectTimer);
			} else if(itemEventElement.attribute("Action") == "popout"){
				//popOutEvents[effectTimer.getLanguage()] = effectTimer;
				popOutEvents.push(effectTimer);
			} else if(itemEventElement.attribute("Action") == "cutin"){
				cutInEvents.push(effectTimer);
				//cutIn();	// alpha 100% + mouse listener
			} else if(itemEventElement.attribute("Action") == "cutout"){
				cutOutEvents.push(effectTimer);
			} else if(itemEventElement.attribute("Action") == "activate"){		// only used for buttons
				activateEvents.push(effectTimer);
			} else if(itemEventElement.attribute("Action") == "deactivate"){	// only used for buttons
				deactivateEvents.push(effectTimer);
			}
		}
		
		protected function parseItemEntries(itemEntryElement:XML){
			var itemEntry:Object = new Object();
			itemEntry["language"] = itemEntryElement.attribute("Language");
			itemEntry["gender"] = itemEntryElement.attribute("Gender");
			itemEntry["age"] = itemEntryElement.attribute("Age");
			itemEntry["value"] = itemEntryElement;
			if(itemType == "Button" || itemType == "ButtonList"){
				itemEntry["action"] = itemEntryElement.attribute("Action");
				itemEntry["link"] = itemEntryElement.attribute("Link");
				
				// add check for MinWidth code here //ds
				if (itemEntryElement.attribute("MinWidth").length() > 0) {
					itemEntry["minWidth"] = itemEntryElement.attribute("MinWidth");
				} else {
					//set it to nothing for check in ButtonItem.as
					itemEntry["minWidth"] = null;
				}
			}
			if(itemType == "Video" || itemType == "Audio"){
				itemEntry["volume"] = Number(itemEntryElement.attribute("Volume"));
				if(itemEntryElement.attribute("Loop") == "true"){
					itemEntry["looping"] = true;
				} else {
					itemEntry["looping"] = false;
				}
			}
				
			if(itemType == "Text"){
				itemEntry["font"] = itemEntryElement.attribute("Font");
				itemEntry["size"] = itemEntryElement.attribute("Size");
				itemEntry["color"] = itemEntryElement.attribute("Color");
				itemEntry["textalign"] = itemEntryElement.attribute("TextAlign");
				
				// have to check strings and assign booleans this war or it doesn't work
				if (itemEntryElement.attribute("Bold").length() > 0) {
					if (itemEntryElement.attribute("Bold") == "true") {
						itemEntry["bold"] = true;
					} else {
						itemEntry["bold"] = false;
					}
				} else {
					itemEntry["bold"] = false;
				}
			
				if (itemEntryElement.attribute("Shadow").length() > 0) {
					if (itemEntryElement.attribute("Shadow") == "true") {
						itemEntry["shadow"] = true;
					} else {
						itemEntry["shadow"] = false;
					}
				} else {
					itemEntry["shadow"] = false;
				}
				
				if (itemEntryElement.attribute("Leading").length() > 0) {
					itemEntry["leading"] = itemEntryElement.attribute("Leading");
				} else {
					itemEntry["leading"] = 0;
				}
				
			}
			
			// check for overriding x/y width/height data
			if(itemEntryElement.attribute("x").length() > 0){
				if(itemEntryElement.attribute("x").indexOf("%", 0) > 0){
					itemEntry["x"] = Number(itemEntryElement.attribute("x").split("%")[0])/100 * conductor.parent.stage.stageWidth;
				} else {
					itemEntry["x"] = Number(itemEntryElement.attribute("x"));
					trace(itemEntry["x"]);
				}
			}
			if(itemEntryElement.attribute("y").length() > 0){
				if(itemEntryElement.attribute("y").indexOf("%", 0) > 0){
					itemEntry["y"] = Number(itemEntryElement.attribute("y").split("%")[0])/100 * conductor.parent.stage.stageHeight;
				} else {
					itemEntry["y"] = Number(itemEntryElement.attribute("y"));
				}
			}
			if(itemEntryElement.attribute("width").length() > 0){
				if(itemEntryElement.attribute("width").indexOf("%", 0) > 0){
					itemEntry["width"] = Number(itemEntryElement.attribute("width").split("%")[0])/100 * conductor.parent.stage.stageWidth;
				} else {
					itemEntry["width"] = Number(itemEntryElement.attribute("width"));
				}
			}
			if(itemEntryElement.attribute("height").length() > 0){
				if(itemEntryElement.attribute("height").indexOf("%", 0) > 0){
					itemEntry["height"] = Number(itemEntryElement.attribute("height").split("%")[0])/100 * conductor.parent.stage.stageHeight;
				} else {
					itemEntry["height"] = Number(itemEntryElement.attribute("height"));
				}
			}
			
			itemEntries[itemEntry["language"]] = itemEntry;
		}
		
		private function removeMouseListener():void{
			removeEventListener(MouseEvent.MOUSE_DOWN, mousePressed);
			clickable = false;
		}
		
		public function reset():void{
			trace("item "+getName()+" RESETTING");
			resetEffectTimers();
			resetTweens();
			removeMouseListener();
			alpha = 0;
		}
		
		public function resetEffectTimers():void{
			// reset all effect timers to prevent stuff from triggering after being removed
			
			for each(var e:EffectTimer in fadeInEvents){
				e.resetTimer();
			}
			for each(e in fadeOutEvents){
				e.resetTimer();
			}
			for each(e in moveInEvents){
				e.resetTimer();
			}
			for each(e in moveOutEvents){
				e.resetTimer();
			}
			for each(e in popInEvents){
				e.resetTimer();
			}
			for each(e in popOutEvents){
				e.resetTimer();
			}
			for each(e in cutInEvents){
				e.resetTimer();
			}
			for each(e in cutOutEvents){
				e.resetTimer();
			}
			for each(e in activateEvents){
				e.resetTimer();
			}
			for each(e in deactivateEvents){
				e.resetTimer();
			}
		}
		
		public function resetTweens():void{
			if(alphaTween != null){
				TweenLite.removeTween(alphaTween);
			}
			if(sizeTween != null){
				TweenLite.removeTween(sizeTween);
			}
			if(posTween != null){
				TweenLite.removeTween(posTween);
			}
		}
		
		private function soundIOError(event:Event):void{
			trace("ERROR: failed to load background audio");
		}
		
		private function soundLoaded(event:Event):void{
			var sound:SoundChannel = soundPlayer.play();
			var soundTrans:SoundTransform = new SoundTransform(conductor.getGain());
			sound.soundTransform = soundTrans;
		}
		
		public function startEffectTimer(events:Array):void{
			var allLanguageEvents:Array = new Array();		// instantiate arrays
			var userLanguageEvents:Array = new Array();
			for each (var e:EffectTimer in events){
				if(e.getLanguage() == "all"){
					allLanguageEvents.push(e);
				}
			}
			if(allLanguageEvents.length > 0){		// if there is at least one "all" entry...
				for each (e in allLanguageEvents){
					//trace("DELAY: "+e.getDelay());
					if(e.isTimedEvent()){
						if(e.getDelay() > 0){
							e.startTimer();
						} else {
							e.startEffect();
						}
					}
				}
			} else {								// use the user language
				for each (e in events){
					if(e.getLanguage() == conductor.getLanguage()){
						userLanguageEvents.push(e);
					}
				}
				if(userLanguageEvents.length > 0){	// if there is at least one user language entry...
					for each (e in userLanguageEvents){
						//trace("DELAY: "+e.getDelay());
						if(e.isTimedEvent()){
							if(e.getDelay() > 0){
								e.startTimer();
							} else {
								e.startEffect();
							}
						}
					}
				}
			}
		}
		
		public function startEffectTimers():void{
			if(fadeInEvents.length == 0 && cutInEvents.length == 0){
				if(itemType == "Audio"){
					//if(startAudioCallback != null){
						startAudioCallback();
					//}
				}
			}
			startEffectTimer(fadeInEvents);
			startEffectTimer(fadeOutEvents);
			startEffectTimer(moveInEvents);
			startEffectTimer(moveOutEvents);
			startEffectTimer(popInEvents);
			startEffectTimer(popOutEvents);
			startEffectTimer(cutInEvents);
			startEffectTimer(cutOutEvents);
			startEffectTimer(activateEvents);
			startEffectTimer(deactivateEvents);
		}
		
		public function triggerVideoEndEffect(events:Array):int{
			var maxDuration:int = 0;
			var allLanguageEvents:Array = new Array();		// instantiate arrays
			var userLanguageEvents:Array = new Array();
			for each (var e:EffectTimer in events){
				if(e.getLanguage() == "all"){
					allLanguageEvents.push(e);
				}
			}
			if(allLanguageEvents.length > 0){		// if there is at least one "all" entry...
				for each (e in allLanguageEvents){
					if(e.isOnVideoEnd()){
						trace("VIDEO END effect called (all languages)");
						if(e.getDuration() > maxDuration){
							maxDuration = e.getDuration();
						}
						e.startEffect();
					}
				}
			} else {								// use the user language
				for each (e in events){
					if(e.getLanguage() == conductor.getLanguage()){
						userLanguageEvents.push(e);
					}
				}
				if(userLanguageEvents.length > 0){	// if there is at least one user language entry...
					for each (e in userLanguageEvents){
						if(e.isOnVideoEnd()){
						trace("VIDEO END effect called (user language)");
							if(e.getDuration() > maxDuration){
								maxDuration = e.getDuration();
							}
							e.startEffect();
						}
					}
				}
			}
			return maxDuration;
		}
		
		public function triggerVideoEndEffects():int{
			// only triggers effects with timers specified "onVideoEnd" for use when a unit closes
			var maxDuration:int = 0;
			var duration:int = 0;
			duration = triggerVideoEndEffect(fadeInEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerVideoEndEffect(moveInEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerVideoEndEffect(popInEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerVideoEndEffect(cutInEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			return maxDuration;
		}
		
		public function triggerExitEffect(events:Array):int{
			var maxDuration:int = 0;
			var allLanguageEvents:Array = new Array();		// instantiate arrays
			var userLanguageEvents:Array = new Array();
			for each (var e:EffectTimer in events){
				if(e.getLanguage() == "all"){
					allLanguageEvents.push(e);
				}
			}
			if(allLanguageEvents.length > 0){		// if there is at least one "all" entry...
				for each (e in allLanguageEvents){
					if(e.isOnExit()){
						if(e.getDuration() > maxDuration){
							maxDuration = e.getDuration();
						}
						e.startEffect();
					}
				}
			} else {								// use the user language
				for each (e in events){
					if(e.getLanguage() == conductor.getLanguage()){
						userLanguageEvents.push(e);
					}
				}
				if(userLanguageEvents.length > 0){	// if there is at least one user language entry...
					for each (e in userLanguageEvents){
						if(e.isOnExit()){
							if(e.getDuration() > maxDuration){
								maxDuration = e.getDuration();
							}
							e.startEffect();
						}
					}
				}
			}
			return maxDuration;
		}
		
		public function triggerExitEffects():int{
			// only triggers effects with timers specified "onExit" for use when a unit closes
			var maxDuration:int = 0;
			var duration:int = 0;
			duration = triggerExitEffect(fadeOutEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerExitEffect(moveOutEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerExitEffect(popOutEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			//trace("trigger exit effects");
			return maxDuration;
		}
		
		public function triggerOnCompleteEffect(events:Array):int{
			var maxDuration:int = 0;
			var allLanguageEvents:Array = new Array();		// instantiate arrays
			var userLanguageEvents:Array = new Array();
			for each (var e:EffectTimer in events){
				if(e.getLanguage() == "all"){
					allLanguageEvents.push(e);
				}
			}
			if(allLanguageEvents.length > 0){		// if there is at least one "all" entry...
				for each (e in allLanguageEvents){
					if(e.isOnComplete()){
						if(e.getDuration() > maxDuration){
							maxDuration = e.getDuration();
						}
						e.startEffect();
					}
				}
			} else {								// use the user language
				for each (e in events){
					if(e.getLanguage() == conductor.getLanguage()){
						userLanguageEvents.push(e);
					}
				}
				if(userLanguageEvents.length > 0){	// if there is at least one user language entry...
					for each (e in userLanguageEvents){
						if(e.isOnComplete()){
							if(e.getDuration() > maxDuration){
								maxDuration = e.getDuration();
							}
							e.startEffect();
						}
					}
				}
			}
			return maxDuration;
		}
		
		public function triggerOnCompleteEffects():int{
			// only triggers effects with timers specified "onComplete" for use when a buttonpairlist
			// has had all of its buttonpairs activated.
			var maxDuration:int = 0;
			var duration:int = 0;
			duration = triggerOnCompleteEffect(fadeInEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerOnCompleteEffect(fadeOutEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerOnCompleteEffect(moveInEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerOnCompleteEffect(moveOutEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerOnCompleteEffect(popInEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerOnCompleteEffect(popOutEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerOnCompleteEffect(activateEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			duration = triggerOnCompleteEffect(deactivateEvents);
			if(duration > maxDuration){
				maxDuration = duration;
			}
			return maxDuration;
			
		}
		
	}
	
}