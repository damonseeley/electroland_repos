package net.electroland.kioskengine  {
	import flash.display.Sprite;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import net.electroland.utils.*;
	
	public class Unit extends Sprite {
		protected var conductor:Conductor;
		private var unitID:String;
		private var unitType:String;
		private var defaultAction:String;
		private var defaultLink:String;
		private var itemList:Array = new Array();
		private var unitDescription:String;
		private var backgroundAudio:String;
		private var audioVolume:int;
		private var delayTimer:Timer;	// used for inactivity screen
		private var exitTimer:Timer;
		private var itemEvent:ItemEvent;
		public var crossFade:Boolean = false;		// default jump cut
		public var crossFadeDuration:int = 2000;
		private var itemsLoaded = 0;
		private var contentLoadedCallback:Function;
		private var loaderTimeOut:Timer;
		private var videoExists:Boolean = false;
		private var videoLooping:Boolean = false;
		private var audioExists:Boolean = false;
		private var mediaFileCount = 0;
		private var mediaFilesPlayed = 0;
		public var ignoreTimeout:Boolean = false;
		public var running = false;
		private var unitLoaded = false;
		private var requester:String;	// used for transitions
		private var resetTimer:Timer;
		private var module:Module;		// reference to module this unit is organized in
		private var contingencies:Array = new Array(); // contingencies to forward to a different unit
		public var startLogger:Boolean = false;
		
		private var hb:HeartBeatTimer;
		
		/*
		
		UNIT.as
		by Aaron Siegel, 7-3-09
		
		Displays all of the objects that extend Item (video, audio, images, text, buttons, etc).
		
		*/
		
		public function Unit(conductor:Conductor, unitXML:XML){
			this.conductor = conductor;
			loaderTimeOut = new Timer(5000,1);	// timer to automatically report unit as loaded
			loaderTimeOut.addEventListener("timer", loaderCallback);
			parseXML(unitXML);
			
			//hb = new HeartBeatTimer("HEARTBEAT unit: " + this.name,3000);
		}
		
		public function addContentListener(f:Function):void{
			 contentLoadedCallback = f;
		}
		
		public function exitUnit(event:TimerEvent):void{
			if(itemEvent != null){
				dispatchEvent(new ItemEvent(itemEvent.getAction(), itemEvent.getArguments()));
			} else {
				var e:ItemEvent = new ItemEvent(defaultAction, defaultLink); 
				e.setSrc(unitID);
				dispatchEvent(e);
			}
		}
		
		public function fadeOutAndExit():void{
			// FADE OUT ITEMS AND USE A TIMER TO TRIGGER DISPATCH OF EVENT
			var maxDuration:int = 0;
			for each (var item in itemList){
				item.resetEffectTimers();
				item.resetTweens();
				var delay:int = item.triggerExitEffects();
				if(delay > maxDuration){
					maxDuration = delay;
				}
			}
			//trace("fade out duration: "+maxDuration);
			if(maxDuration > 0){
				exitTimer = new Timer(maxDuration, 1);	// duration should be based on longest exit effect
				exitTimer.addEventListener("timer", exitUnit);
				exitTimer.start();
			} else {
				if(itemEvent.getSrc() != null){
					var newEvent = new ItemEvent(itemEvent.getAction(), itemEvent.getArguments());
					newEvent.setSrc(itemEvent.getSrc());
					dispatchEvent(newEvent);
				} else {
					dispatchEvent(new ItemEvent(itemEvent.getAction(), itemEvent.getArguments()));
				}
			}
		}
		
		public function getName():String{
			return unitID;
		}
		
		public function getDescription():String{
			return unitDescription;
		}
		
		public function getType():String{
			return unitType;
		}
		
		public function getModule():Module{
			return module;
		}
		
		public function setModule(module:Module):void{
			this.module = module;
		}
		
		public function itemEventHandler(event:ItemEvent):void{
			//trace(event.getAction() +" "+ event.getArguments());
			if(event.getAction() == "unit" || event.getAction() == "module"){
				// if going to a new unit or module, report back to conductor
				//dispatchEvent(new ItemEvent(event.getAction(), event.getArguments()));
				itemEvent = event;	// stored until exitTimer is triggered
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				trace("SRC SHOULD BE: "+itemEvent.getSrc());
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "play" || event.getAction() == "pause"){
				// receives an event to launch a script or something within this unit
				for each (var item in itemList){
					if(item.getName() == event.getArguments()){
						if(event.getAction() == "play"){
							item.play();
						} else {
							item.pause();
						}
					}
				}
			} else if(event.getAction() == "setLanguage"){
				conductor.setLanguage(event.getArguments());
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "setGender"){
				conductor.setGender(event.getArguments());
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "setAge"){
				conductor.setAge(event.getArguments());
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "videoComplete"){	// if video finished playing...
				if(unitType == "video"){						// and this is a passive video unit...
					itemEvent = new ItemEvent(defaultAction, defaultLink);
					itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
					conductor.resetInactivityTimer();
					fadeOutAndExit();
				} else {
					/*
					if(!videoLooping){
						if(!ignoreTimeout){
							conductor.startInactivityTimer();		// start inactivity timer on videoComplete
						}
					}
					*/
					mediaFilesPlayed++;
					if(unitType == "interactive"){
						if(mediaFileCount == mediaFilesPlayed){
							trace("######> unit "+unitID+" calling inactivity timer start cause all media has played");
							conductor.startInactivityTimer();
						}
					}
					// check items for videoEnd event
					for each (item in itemList){
						item.triggerVideoEndEffects();
					}
				}
			} else if(event.getAction() == "audioComplete"){	// if audio finished playing...
				if(unitType == "audio"){						// and this is a passive audio unit...
					itemEvent = new ItemEvent(defaultAction, defaultLink);
					itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
					conductor.resetInactivityTimer();
					fadeOutAndExit();
				} else {
					mediaFilesPlayed++;
					if(unitType == "interactive"){
						if(mediaFileCount == mediaFilesPlayed){
							trace("######> unit "+unitID+" calling inactivity timer start cause all media has played");
							conductor.startInactivityTimer();
						}
					}
				}				
			} else if(event.getAction() == "default"){
				// follow the default action/link of this unit
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "print"){
				// execute the command to print a document here
				trace("PRINT COMMAND RECEIVED");
				conductor.print(event.getArguments());
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "wakeup"){
				itemEvent = event;	// stored until exitTimer is triggered
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "resetSession"){
				itemEvent = event;	// stored until exitTimer is triggered
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "onComplete"){
				// for button list item, should trigger onComplete effects
				for each (item in itemList){
					item.triggerOnCompleteEffects();
				}
			} else if(event.getAction() == "setReviewQuestionOne"){
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "setReviewQuestionTwo"){
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "setReviewQuestionThree"){
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "setReviewQuestionFour"){
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else if(event.getAction() == "setReviewQuestionFive"){
				itemEvent = new ItemEvent(defaultAction, defaultLink);
				itemEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				conductor.resetInactivityTimer();
				fadeOutAndExit();
			} else {	// doesn't match any condition
				var newEvent:ItemEvent = new ItemEvent(event.getAction(), event.getArguments());
				newEvent.setSrc(unitID);	// tell conductor that this command is from this unit
				dispatchEvent(newEvent);
			}
			
			conductor.logEvent(event);	// logging EVERYTHING
		}
		
		public function itemLoaded(item:Item):void{
			trace(getName()+".itemLoaded(): "+ item.getName() +" LOADED");
			itemsLoaded++;
			if(contentLoadedCallback != null){		// make sure there is a listener established
				//if(itemsLoaded == 1){				// kludge for now since image is the only item doing the callback
				if(itemsLoaded == itemList.length){
					loaderTimeOut.reset();
					unitLoaded = true;
					
					// check for contingencies before telling the conductor to switch to this unit
					var keepGoing:Boolean = true;
					var winningContingency:Object;
					if(contingencies.length > 0){
						// loop through items, find all external buttons, and count how many are deactivated
						var numDeactivated:int = 0;
						for each (var item:Item in itemList){
							if(item.deactivated){
								numDeactivated++;
							}
						}
						for each (var contingency:Object in contingencies){		// priority of contingency depends on order in xml
							if(contingency["type"] == "ButtonContingency"){
								if(contingency["totalButtons"] <= numDeactivated){	// qualifies for contingency
									keepGoing = false;		// prevent unit from telling the conductor to start it
									winningContingency = contingency;
								}
							}
						}
					}
					
					if(keepGoing){
						// inform the conductor this unit has been fully loaded	
						contentLoadedCallback(this, requester);	// return this obj and the name of the unit prior to it
					} else {
						// inform the conductor to go to do the action/link from this contingency
						itemEvent = new ItemEvent(winningContingency["action"], winningContingency["link"]);
						itemEvent.setSrc(requester);	// src is the unit that requested this unit
						dispatchEvent(itemEvent);
						reset();	// immediately reset, as this unit has NOT been added to the stage
					}
				}
			}
		}
		
		public function loaderCallback(event:TimerEvent):void{
			// not all items reported back, timeout delay has been hit, so forcing callback that this unit has loaded.
			trace("ERROR: UNIT "+getName()+ " FAILED TO LOAD ALL ITEMS");
			if(contentLoadedCallback != null){
				contentLoadedCallback(this, requester);
			}
		}
		
		public function parseXML(unitElement:XML):void{
			unitID = unitElement.attribute("UnitID");
			unitType = unitElement.attribute("UnitType");
			unitDescription = unitElement.Description;
			if(unitElement.hasOwnProperty("Transition")){
				crossFade = true;
				crossFadeDuration = Number(unitElement.Transition.attribute("Duration"));
			}
			if(unitElement.attribute("DefaultAction").length() > 0){
				defaultAction = unitElement.attribute("DefaultAction");
			}
			if(unitElement.attribute("DefaultLink").length() > 0){
				defaultLink = unitElement.attribute("DefaultLink");
			}
			if(unitElement.attribute("StartLogger").length() > 0){
				startLogger = true;
			}
			if(unitElement.attribute("Timer").length() > 0){
				delayTimer = new Timer(Number(unitElement.attribute("Timer")), 1);
				delayTimer.addEventListener("timer", exitUnit);	// dispatch event to parent and skip exit effects
			}
			if(unitElement.hasOwnProperty("BackgroundAudio")){
				backgroundAudio = unitElement.BackgroundAudio;
				audioVolume = Number(unitElement.BackgroundAudio.attribute("VolumeLevel"));
			}
			if(unitElement.hasOwnProperty("Contingencies")){
				for each (var buttonContingencyElement:XML in unitElement.Contingencies.ButtonContingency){
					var buttonContingency:Object = new Object();
					buttonContingency["type"] = "ButtonContingency";
					buttonContingency["totalButtons"] = buttonContingencyElement.attribute("TotalButtons");
					buttonContingency["action"] = buttonContingencyElement.attribute("Action");
					buttonContingency["link"] = buttonContingencyElement.attribute("Link");
					contingencies.push(buttonContingency);
				}
			}
			for each (var itemElement:XML in unitElement.Items.Item){
				//trace(itemElement.attribute("Type"));	// use type to determine what object to create
				if(itemElement.attribute("Type") == "Button"){
					var button:ButtonItem = new ButtonItem(conductor, itemElement);
					itemList.push(button);
				} else if(itemElement.attribute("Type") == "Image"){
					var image:ImageItem = new ImageItem(conductor, itemElement);
					itemList.push(image);
				} else if(itemElement.attribute("Type") == "Video"){
					var video:VideoItem = new VideoItem(conductor, itemElement);
					itemList.push(video);
				} else if(itemElement.attribute("Type") == "Audio"){
					var audio:AudioItem = new AudioItem(conductor, itemElement);
					itemList.push(audio);
				} else if(itemElement.attribute("Type") == "ButtonList"){
					var buttonList:ButtonListItem = new ButtonListItem(conductor, itemElement);
					itemList.push(buttonList);
				} else if(itemElement.attribute("Type") == "ButtonPairList"){
					var buttonPairList:ButtonPairListItem = new ButtonPairListItem(conductor, itemElement);
					itemList.push(buttonPairList);
				} else if(itemElement.attribute("Type") == "Text"){
					var text:TextItem = new TextItem(conductor, itemElement);
					itemList.push(text);
				} else if(itemElement.attribute("Type") == "TextList"){
					
				}
			}
		}
		
		public function delayedReset(duration:int):void{
			trace("delayed reset of: "+duration+"ms");
			resetTimer = new Timer(duration,1);
			resetTimer.addEventListener("timer", timedReset);
			resetTimer.start();
			// loop through items for audio and video, and mute their sound immediately
			for each (var item in itemList){
				if(item.getType() == "Audio" || item.getType() == "Video"){
					trace("MUTING "+item.getName());
					item.mute();
				}
			}
		}
		
		public function timedReset(event:TimerEvent):void{
			reset();
		}
		
		public function reset():void{
			// reset all the item timers in this unit
			trace("unit "+getName()+" received RESET call");
			for each (var item in itemList){
				if(contains(item)){
					removeChild(item);
					trace("item "+ item.getName() + " REMOVED from stage");
					item.reset();
					item.removeEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
				}
			}
			running = false;
			unitLoaded = false;
			itemsLoaded = 0;
			dispatchEvent(new ItemEvent("removeFromStage", unitID));
		}
		
		public function loadContentAndReturn(requester:String):void{
			this.requester = requester;
			loadContent();
		}
		
		public function loadContent():void{
			if(!unitLoaded){				
				// load everything before start() gets called
				loaderTimeOut.start();	// timeout if not all items report back
				trace("LOADER TIMEOUT STARTED");
				for each (var item in itemList){
					item.addContentListener(itemLoaded);
					item.loadContent();
					//addChild(item);
					//item.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
				}
			}
		}
		
		public function start():void{
			if(!running){
				// begin timer events for all items to be displayed on screen.
				videoExists = false;
				audioExists = false;
				videoLooping = false;
				for each (var item in itemList){
					item.startTimer();
					if(item.getType() == "Video"){							// check if item is a video
						videoExists = true;									// video exists
						mediaFileCount++;
						var itemEntries:Object = item.getItemEntries();		// check if video is looping...
						if(itemEntries.hasOwnProperty("all")){
							if(itemEntries["all"].looping){
								videoLooping = true;
							}
						} else if(itemEntries.hasOwnProperty(conductor.getLanguage())){
							if(itemEntries[conductor.getLanguage()].looping){
								videoLooping = true;
							}
						}
					} else if(item.getType() == "Audio"){
						audioExists = true;
						mediaFileCount++;
					}
					item.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
					addChild(item);
					trace(item.getName()+" alpha: "+item.alpha);
				}
				conductor.setBackgroundAudio(backgroundAudio, audioVolume);
				if(unitType == "interactive"){
					if(!videoExists && !audioExists){
						if(!ignoreTimeout){
							trace("######> unit "+unitID+" calling inactivity timer start on unit start");
							conductor.startInactivityTimer();
						}
					}
				}
				if(delayTimer != null){
					delayTimer.start();
				}
				running = true;
			}
		}
		
	}
	
}