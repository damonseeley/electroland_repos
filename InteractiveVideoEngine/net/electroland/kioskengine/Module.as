package net.electroland.kioskengine  {
	
	public class Module {
		private var conductor:Conductor;
		private var moduleName:String
		private var moduleDescription:String;
		private var logVariable:int;
		private var entryPoints:Array = new Array();
		private var unitNames:Array = new Array();	// strings with names of units in this module
		
		/*
		
		MODULE.AS
		by Aaron Siegel, 8-27-09
		
		This object does nothing but contain entry points for the module and returns
		the one pertinent to the users selected	language, gender, and age group.
		
		*/
		
		public function Module(conductor:Conductor, moduleXML:XML){
			this.conductor = conductor;
			parseXML(moduleXML);
		}
		
		public function getEntryPoint():String{
			if(entryPoints.length > 1){
				var multiLingual:Array = new Array();
				var usersLanguage:Array = new Array();
				for each(var entryPoint in entryPoints){
					if(entryPoint["Language"] == "all"){	// check everything for Language == "all"
						multiLingual.push(entryPoint);
					}
				}
				if(multiLingual.length > 0){
					for each(entryPoint in multiLingual){
						if(entryPoint.hasOwnProperty("Gender")){				// check against user gender
							if(entryPoint["Gender"] == conductor.getGender()){	// language = "all" and gender is this users
								return entryPoint["FirstUnit"];
							}
						}
					}
				} else {									// no "all", so look for user language
					for each(entryPoint in entryPoints){
						if(entryPoint["Language"] == conductor.getLanguage()){
							usersLanguage.push(entryPoint);
						}
					}
					for each(entryPoint in usersLanguage){
						if(entryPoint.hasOwnProperty("Gender")){				// check against user gender
							if(entryPoint["Gender"] == conductor.getGender()){	// language is users and gender is users
								return entryPoint["FirstUnit"];
							}
						}
					}
				}
			}
			return entryPoints[0]["FirstUnit"];			// return the only entry
		}
		
		public function getName():String{
			return moduleName;
		}
		
		public function getDescription():String{
			return moduleDescription;
		}
		
		public function getUnitNames():Array{
			return unitNames;
		}
		
		private function parseXML(moduleElement:XML):void{
			logVariable = Number(moduleElement.attribute("LogVariable"));			// log variable not actually used
			moduleName = moduleElement.ModuleName.valueOf();						// module name
			moduleDescription = moduleElement.Description.valueOf();				// module description
			for each (var entryPoint:XML in moduleElement.EntryPoints.EntryPoint){	// for each entry point...
				var obj:Object = new Object();
				for each (var attribute in entryPoint.attributes()){				// grab each attribute
					obj[String(attribute.name())] = attribute;						// store attributes in an object
				}
				entryPoints.push(obj);												// store objects in an array
			}
			var unitXmlList:XMLList = moduleElement.Units.Unit;
			for each (var unitElement:XML in unitXmlList) {
				unitNames.push(unitElement.attribute("UnitID"));
			}
		}
		
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*
	import flash.display.Sprite;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import gs.*;
	
	public class Module extends Sprite {
		private var conductor:Conductor;
		private var moduleName:String
		private var moduleDescription:String;
		private var logVariable:int;
		private var currentUnit:Unit;				// unit currently running
		private var nextUnit:Unit;					// unit crossfading into
		private var unitList:Object = new Object();	// actual unit objects addressed by name
		private var unitNames:Array = new Array();	// strings with names of units in this module
		//private var entryPoints:Object = new Object();
		private var entryPoints:Array = new Array();
		private var exitTimer:Timer;
		public var crossFade:Boolean = false;		// default jump cut
		public var crossFadeDuration:int = 2000;
		private var contentLoadedCallback:Function;
		
		
	
//		MODULE.AS
//		by Aaron Siegel, 7-3-09
//		
//		This object displays units which contain lists of items to display such as
//		video objects, text, and buttons. The module should be treated as a list of
//		units and only exists to control their instantiation, destruction, and what
//		unit to display next based on the users input.	
		
		
		public function Module(conductor:Conductor, moduleXML:XML){
			this.conductor = conductor;
			parseXML(moduleXML);
		}
		
		public function addContentListener(f:Function):void{
			 contentLoadedCallback = f;
		}
		
		public function getName():String{
			return moduleName;
		}
		
		public function getDescription():String{
			return moduleDescription;
		}
		
		public function getLogID():int{
			return logVariable;
		}
		
		public function getCurrentUnit():Unit{
			return currentUnit;
		}
		
		public function getUnitNames():Array{
			return unitNames;
		}
		
		
//		public function crossFadeUnits(targetUnit:String, duration:int):void{
//			nextUnit = unitList[targetUnit];
//			nextUnit.alpha = 0;
//			nextUnit.addContentListener(unitLoaded);
//			//nextUnit.start();
//			//addChild(nextUnit);
//			conductor.logEvent(new ItemEvent("unit", targetUnit));
//			conductor.updateDebugDisplay(this, nextUnit);
//			exitTimer = new Timer(duration, 1);	// duration should be based on crossfade duration
//			exitTimer.addEventListener("timer", removeUnit);
//			exitTimer.start();
//			TweenLite.to(nextUnit, duration/1000, {alpha:1});	// same duration as crossfade
//		}
		
		
		public function removeUnit(event:TimerEvent):void{
			//trace("-------------> REMOVE UNIT TIMER EVENT");
			//if(getChildByName(currentUnit.getName())){
				removeChild(currentUnit);
				trace("unit "+currentUnit.getName()+" REMOVED from stage (timer event)");
				currentUnit.reset();
				currentUnit = nextUnit;
				//trace("nextUnit alpha: "+nextUnit.alpha);
				nextUnit = null;
			//}
		}
		
		public function itemEventHandler(event:ItemEvent):void{
			if(event.getAction() == "module"){
				// set currentUnit as have been viewed
				conductor.setUnitViewed(currentUnit.getName());
				// set currentModule as have been viewed
				conductor.setModuleViewed(moduleName);
				// if going to a new module, report back to conductor
				dispatchEvent(new ItemEvent(event.getAction(), event.getArguments()));
			} else if (event.getAction() == "unit"){
				// receives events to switch units within this module
				//trace(event.getAction() +" "+ event.getArguments());
				
				// set currentUnit as have been viewed
				conductor.setUnitViewed(currentUnit.getName());
				
				
//				if(unitList[event.getArguments()].crossFade){
//					crossFadeUnits(event.getArguments(), unitList[event.getArguments()].crossFadeDuration);
//				} else {
//					jumpToUnit(event.getArguments());
//				}
				jumpToUnit(event.getArguments());
			}
		}
		
		public function jumpToUnit(unitName:String):void{
			nextUnit = unitList[unitName];
			nextUnit.addContentListener(unitLoaded);
			nextUnit.loadContent();
		}
		
		public function parseXML(moduleElement:XML){
			logVariable = Number(moduleElement.attribute("LogVariable"));
			moduleName = moduleElement.ModuleName.valueOf();
			moduleDescription = moduleElement.Description.valueOf();
			if(moduleElement.hasOwnProperty("Transition")){
				crossFade = true;
				crossFadeDuration = Number(moduleElement.Transition.attribute("Duration"));
			}
			for each (var entryPoint:XML in moduleElement.EntryPoints.EntryPoint){
				var obj:Object = new Object();
				for each (var attribute in entryPoint.attributes()){
					obj[String(attribute.name())] = attribute;
					//trace(attribute.name() +" "+ attribute);
				}
				entryPoints.push(obj);
				//entryPoints[entryPoint.attribute("Age")] = entryPoint.attribute("FirstUnit");
			}
			var unitXmlList:XMLList = moduleElement.Units.Unit;
			for each (var unitElement:XML in unitXmlList) {
				// create a new unit and pass it the unitElement
				var unit:Unit = new Unit(conductor, unitElement);
				unit.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
				conductor.addToUnitStates(unit.getName());
				unitList[unit.getName()] = unit;
				unitNames.push(unit.getName());
			}
		}
		
		public function loadContent():void{
			//trace("Module "+ moduleName +" started");
			// start unit as specified by entry points
			if(entryPoints.length > 1){
				var multiLingual:Array = new Array();
				var usersLanguage:Array = new Array();
				for each(var entryPoint in entryPoints){
					// check everything for Language == "all"
					if(entryPoint["Language"] == "all"){
						multiLingual.push(entryPoint);
					}
				}
				if(multiLingual.length > 0){
					//trace("multilingual entry points: "+multiLingual.length);
					for each(entryPoint in multiLingual){
						// check against user gender
						if(entryPoint.hasOwnProperty("Gender")){
							if(entryPoint["Gender"] == conductor.getGender()){
								// language = "all" and gender is this users
								nextUnit = unitList[entryPoint["FirstUnit"]];
								nextUnit.addContentListener(unitLoaded);
								nextUnit.loadContent();
								//addChild(currentUnit);		// these two lines should happen after
								//currentUnit.start();		// unit calls back that content has loaded.
							}
						}
					}
				} else {
					// no "all", so look for user language
					for each(entryPoint in entryPoints){
						if(entryPoint["Language"] == conductor.getLanguage()){
							usersLanguage.push(entryPoint);
						}
					}
					for each(entryPoint in usersLanguage){
						// check against user gender
						if(entryPoint.hasOwnProperty("Gender")){
							if(entryPoint["Gender"] == conductor.getGender()){
								// language is users and gender is users
								nextUnit = unitList[entryPoint["FirstUnit"]];
								nextUnit.addContentListener(unitLoaded);
								nextUnit.loadContent();
								//addChild(currentUnit);		// these two lines should happen after
								//currentUnit.start();		// unit calls back that content has loaded.
							}
						}
					}
				}
			} else {
				nextUnit = unitList[entryPoints[0]["FirstUnit"]];
				nextUnit.addContentListener(unitLoaded);
				nextUnit.loadContent();
				//addChild(currentUnit);		// these two lines should happen after
				//currentUnit.start();		// unit calls back that content has loaded.
			}
			
			//trace(currentUnit);
			//conductor.updateDebugDisplay(this, nextUnit);
		}
		
		public function reset():void{
			// reset all the units to reset all their item timers
			trace("module "+getName()+" received RESET");
			currentUnit.reset();
			currentUnit = null;
		}
		
		public function start():void{
			// should only be run after getting a module loaded callback
			currentUnit.start();
			addChild(currentUnit);
			//conductor.logEvent(new ItemEvent("unit", currentUnit.getName()));
			conductor.logUnitEvent(getName(), currentUnit.getName(), "started");
			conductor.updateDebugDisplay(this, currentUnit);
			trace(currentUnit.getName() +" ADDED to stage");
		}
		
		public function unitLoaded(unit:Unit):void{
			trace("UNIT "+ unit.getName() + " LOADED");
			var firstUnit = false;
			if(unit.crossFade){			// if crossfading...
				trace("-------> CROSSFADE");
				if(currentUnit != null){
					exitTimer = new Timer(unit.crossFadeDuration, 1);	// wait for crossfade before removing old unit
					exitTimer.addEventListener("timer", removeUnit);
					exitTimer.start();
					addChild(unit);
					unit.alpha = 0;
					conductor.logUnitEvent(getName(), unit.getName(), "started");
					unit.start();
					conductor.updateDebugDisplay(this, unit);
				} else {
					firstUnit = true;
					currentUnit = unit;			// set new unit to current unit
				}
				// this tween is running regardless, since it does the visual action
				// when crossfading between modules.
				//trace("nextUnit alpha: "+nextUnit.alpha);
				TweenLite.to(unit, unit.crossFadeDuration/1000, {alpha:1});	// same duration as crossfade
			} else {					// if direct cut...
				trace("-------> DIRECT CUT");
				if(currentUnit != null){		// if not first unit...

//					removeChild(currentUnit);	// remove what's there
//					trace("unit "+currentUnit.getName()+" REMOVED from stage");
//					currentUnit.reset();
//					currentUnit = unit;			// set new unit to current unit
//					currentUnit.start();
//					conductor.logEvent(new ItemEvent("unit", currentUnit.getName()));
//					conductor.updateDebugDisplay(this, currentUnit);
//					addChild(currentUnit);
					
					conductor.logUnitEvent(getName(), unit.getName(), "started");
					unit.start();
					addChild(unit);
					removeChild(currentUnit);
					currentUnit.reset();
					currentUnit = unit;
					//conductor.logEvent(new ItemEvent("unit", currentUnit.getName()));
					conductor.updateDebugDisplay(this, currentUnit);
					
				} else {
					firstUnit = true;
					currentUnit = unit;			// set new unit to current unit
				}
			}
			if(firstUnit){
				// inform the conductor that the first unit has loaded it's content.
				// the first unit will be started when the module is started by the conductor.
				if(contentLoadedCallback != null){
					contentLoadedCallback(this);
				}
			}
		}
		
	}
	*/
	
}