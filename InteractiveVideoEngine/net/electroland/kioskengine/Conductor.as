package net.electroland.kioskengine{

	import flash.display.Sprite;
	import flash.display.MovieClip;
	import flash.events.*;
	import flash.display.Loader;
	import flash.net.URLLoader;
	import flash.net.URLRequest;
	import flash.system.ApplicationDomain;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	//import flash.filesystem.*;
	import flash.net.XMLSocket;
	import gs.*;

	/*
	
	CONDUCTOR.as
	by Aaron Siegel, 7-3-09
	
	Manages which module is active, as well as receiving global properties
	such as background audio changes, user language changes, and activity timeouts.
	
	*/

	public class Conductor extends Sprite {
		//private var currentModule:Module;
		//private var nextModule:Module;
		private var currentUnit:Unit;					// used ONLY as a reference for debug display/logging
		private var modules:Object = new Object();		// assoc. array of module objects addressed by name
		private var moduleNames:Array = new Array();	// strings of module names
		private var units:Object = new Object();		// assoc. array of unit objects addressed by name
		private var unitNames:Array = new Array();		// strings of unit names
		private var userLanguage:String;
		private var userGender:String;
		private var userAge:String;
		private var backgroundAudio:BackgroundAudio;
		private var logger:Logger;
		private var xmlData:XML;
		private var xmlLoader:URLLoader;
		private var xmlFile:String;
		private var debugDisplay:DebugDisplay;
		private var buttonSound:String;
		private var xmlLoaded:Boolean = false;
		private var exitTimer:Timer;
		private var moduleStates:Object = new Object();	// associative array of module names and true/false state
		private var unitStates:Object = new Object();	// associative array of unit names and true/false state
		private var editMode:Boolean = false;			// used for moving objects around and reporting their X/Y pos
		private var inactivityTimer:Timer;
		private var inactivityScreen:Unit;				// used to warn of session timeout
		private var main:Main;							// reference to document class
		private var sessionTimestamp:Date;				// point when session began
		private var sessionID:int = 0;					// iterating index of sessions since kiosk installation
		private var kioskID:String;						// identifies the kiosk in the log
		private var loggerURL:String;					// URL OF REMOTE LOGGER FOLDER (not directly to php file)
		private var printSocket:XMLSocket;
		private var printFile:String;
		private var getNewSessionID:Boolean = true;		// kludge solution for retrieving/retaining sessionID across module XML files
		private var gain:Number = 1.0;					// floating point number to increase/decrease audio item volume

		public function Conductor(main:Main, debugDisplay:DebugDisplay) {
			this.main = main;
			this.debugDisplay = debugDisplay;
			userLanguage = "";// defaults
			userGender = "";
			userAge = "";
			backgroundAudio = new BackgroundAudio(this);
			logger = new Logger();
			printSocket = new XMLSocket();
			printSocket.addEventListener(Event.CONNECT, socketConnectHandler);
			printSocket.addEventListener(IOErrorEvent.IO_ERROR, socketIOErrorHandler);
			printSocket.addEventListener(SecurityErrorEvent.SECURITY_ERROR, securityErrorHandler);
		}
		
		function addInactivityScreen(inactivityScreen:Unit):void{
			this.inactivityScreen = inactivityScreen;
		}
		
		function addToModuleStates(moduleName:String):void {
			moduleStates[moduleName] = false;
			//trace("module "+moduleName+" added");
		}
		
		function checkModuleState(moduleName:String):Boolean {
			if (moduleStates.hasOwnProperty(moduleName)) {
				return moduleStates[moduleName];
			}
			return false;
		}
		
		function setModuleViewed(moduleName:String):void {
			moduleStates[moduleName] = true;
			//trace(moduleName +" was just viewed");
		}
		
		function addToUnitStates(unitName:String):void {
			unitStates[unitName] = false;
			//trace("unit "+unitName+" added");
		}
		
		function checkUnitState(unitName:String):Boolean {
			if (unitStates.hasOwnProperty(unitName)) {
				return unitStates[unitName];
			}
			return false;
		}
		
		function setUnitViewed(unitName:String):void {
			unitStates[unitName] = true;
			//trace(unitName +" was just viewed");
		}
		
		function setInactivityTimer(duration:int){
			inactivityTimer = new Timer(duration,1);
			inactivityTimer.addEventListener("timer", inactivityTimeout);
		}
		
//		function removeModule(event:TimerEvent):void {
//			removeChild(currentModule);
//			trace("module "+ currentModule +" REMOVED from stage (timer event)");
//			currentModule.reset();
//			currentModule = nextModule;
//			//trace("module removed");
//			//trace("alpha: "+nextModule.alpha);
//			nextModule = null;
//		}
		
		public function getAge():String {
			if(userAge == ""){
				return "Young"
			}
			return userAge;
		}
		
		public function getGain():Number{
			return gain;
		}
		
		public function getGender():String {
			if(userGender == ""){
				return "Male";	// default content when no gender has been selected
			}
			return userGender;
		}
		
		public function getLanguage():String {
			if(userLanguage == ""){
				return "English";	// default content when no language has been selected
			}
			return userLanguage;
		}
		
		public function getButtonSound():String {
			return buttonSound;
		}
		
		public function getCurrentModule():Module {
			if(currentUnit != null){
				return currentUnit.getModule();
			}
			return null;
		}
		
		public function getModuleNames():Array {
			return moduleNames;
		}
		
		public function getModule(moduleName:String):Module{
			return modules[moduleName];
		}
		
		public function getUnitNames():Array {
			return unitNames;
		}
		
		public function getSessionDuration():String {
			var sysTime:Date = new Date();
			var diff:int = sysTime.getTime() - sessionTimestamp.getTime();
			var hours:int = diff / (60 * 60 * 1000);
			var remainder:int = diff % (60 * 60 * 1000);
			var minutes:int = remainder / (60 * 1000);
			remainder = remainder % (60 * 1000);
			var seconds:int = remainder / 1000;
			var millis:int = remainder % 1000;
			var sessionDuration:String = String(hours)+":"+String(minutes)+":"+String(seconds);//+":"+String(millis);
			return sessionDuration;
		}
		
		public function isEditMode():Boolean{
			return editMode;
		}
		
		public function setButtonSound(buttonSound):void {
			this.buttonSound = buttonSound;
		}
		
		public function volumeUp(){		// adds 10%
			gain += 0.2;
		}
		
		public function volumeDown(){	// decreases 10%
			gain -= 0.2;
		}
		
		public function itemEventHandler(event:ItemEvent):void {
			// receives events to change modules
			trace("ITEM EVENT: "+ event.getAction() +" "+ event.getArguments());
			//jumpToModule(event.getArguments());	// DEPRECATED
			var unit:Unit;
			if(event.getAction() == "module"){
				trace("old unit: "+event.getSrc() + " new unit: "+ modules[event.getArguments()].getEntryPoint());
				unit = units[modules[event.getArguments()].getEntryPoint()];
				unit.addContentListener(unitLoaded);
				unit.loadContentAndReturn(event.getSrc());
			} else if(event.getAction() == "unit"){
				trace("old unit: "+event.getSrc() + " new unit: "+ event.getArguments());
				unit = units[event.getArguments()];
				unit.addContentListener(unitLoaded);
				unit.loadContentAndReturn(event.getSrc());
			} else if(event.getAction() == "removeFromStage"){
				if(contains(units[event.getArguments()])){
					trace("REMOVING "+event.getArguments()+" FROM THE STAGE");
					removeChild(units[event.getArguments()]);
					setUnitViewed(event.getArguments());
				}
			} else {
				// pass back to Main
				dispatchEvent(new ItemEvent(event.getAction(), event.getArguments()));
			}
		}
		
		public function jumpToModule(moduleName:String):void {
			if (modules.hasOwnProperty(moduleName)) {
				jumpToUnit(modules[moduleName].getEntryPoint());
			}
		}
		
		public function jumpToUnit(unitName:String):void{
			if(units.hasOwnProperty(unitName)){
				var unit:Unit = units[unitName];
				unit.addContentListener(unitLoaded);
				unit.loadContentAndReturn(currentUnit.getName());
			}
		}
		
		public function load(filename:String):void {
			xmlFile = filename;
			xmlData = new XML();
			xmlLoader = new URLLoader();
			xmlLoader.addEventListener(Event.COMPLETE, parseXML);
			xmlLoader.load(new URLRequest(filename));
		}
		
		public function loadModules(filename:String):void{
			// clear current modules/units, retain user data and sessionID, load new modules xml
			for(var n:Number=0; n<numChildren; n++){		// for everything on the stage...
				var oldUnit:Unit = getChildAt(n) as Unit;
				if(oldUnit != null){
					oldUnit.reset();	// might call back to reset if not destroyed immediately
					if(contains(oldUnit)){
						removeChild(oldUnit);
					}
				}
			}
			for(var i:Number=0; i<modules.length; i++){
				var mod:Module = modules[i];
				mod = null;		// destroy module completely
			}
			modules = new Object();		// actual module objects addressed by name
			moduleNames = new Array();	// strings of module names
			for(i=0; i<units.length; i++){
				var unit:Unit = units[i];
				unit = null;	// destroy unit completely
			}
			units = new Object();
			unitNames = new Array();
			xmlFile = filename;
			xmlLoader.load(new URLRequest(xmlFile));
		}
		
		public function logEvent(event:ItemEvent):void {
			// all terminating itemEventHandlers will call this for logging
			//logger.logEvent(event);
		}
		
		public function logUnitEvent(moduleName:String, unitName:String, eventDescription:String):void {
			// system timestamp, session timestamp, kiosk ID#, session ID#, language, sex, age group, event type, module name, unit name, item name, item description, event description
			var sysTime:Date = new Date();
			var sessionDuration:String = getSessionDuration();
			logger.logEvent(sysTime.toString(), sessionDuration, kioskID, String(sessionID), userLanguage, userGender, userAge, "unit", moduleName, unitName, " ", " ", eventDescription);
		}
		
		public function logItemEvent(moduleName:String, unitName:String, itemName:String, itemDescription:String, eventDescription:String):void {
			// system timestamp, session timestamp, kiosk ID#, session ID#, language, sex, age group, event type, module name, unit name, item name, item description, event description
			var sysTime:Date = new Date(); 
			var sessionDuration:String = getSessionDuration();
			logger.logEvent(sysTime.toString(), sessionDuration, kioskID, String(sessionID), userLanguage, userGender, userAge, "item", moduleName, unitName, itemName, itemDescription, eventDescription);
		}
		
//		public function moduleLoaded(m:Module):void {
//			trace("module "+ m.getName() +" loaded");
//			var oldModule:Module;
//			// check module's current unit to see if it's crossfading
//			if (currentModule != null) {
//				oldModule = currentModule;
//			} else {
//				trace("CURRENT MODULE has yet to be instantiated");
//				currentModule = m;
//			}
//			addChild(m);
//			trace(m.getName() + " ADDED to stage");
//			m.start();
//			if (oldModule != null) {
//				if(m.getCurrentUnit().crossFade){
//					exitTimer = new Timer(m.getCurrentUnit().crossFadeDuration, 1);// duration should be based on crossfade duration
//					exitTimer.addEventListener("timer", removeModule);
//					exitTimer.start();
//					trace("module "+oldModule.getName()+" starting exit delay timer of "+m.getCurrentUnit().crossFadeDuration+"ms");
//				} else {
//					removeChild(oldModule);
//					trace(oldModule.getName() +" REMOVED from stage without delay");
//					oldModule.reset();
//					currentModule = m;
//				}
//			}
//		}
		
		public function parseXML(e:Event):void {
			xmlData = new XML(e.target.data);
			var moduleXmlList:XMLList = xmlData.Module;
			for each (var moduleElement:XML in moduleXmlList) {
				// create empty module and pass it the moduleElement
				var module:Module = new Module(this, moduleElement);
				addToModuleStates(module.getName());
				modules[module.getName()] = module;
				moduleNames.push(module.getName());
				
				var unitXmlList:XMLList = moduleElement.Units.Unit;
				for each (var unitElement:XML in unitXmlList) {
					var unit:Unit = new Unit(this, unitElement);		// create a new unit and pass it the unitElement
					unit.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
					unit.setModule(module);
					addToUnitStates(unit.getName());
					units[unit.getName()] = unit;
					unitNames.push(unit.getName());
				}
				
			}
			trace("starting initial unit");
			debugDisplay.addConductor(this);
			start(xmlData.attribute("EntryPoint"));
			xmlLoaded = true;
		}
		
		public function setBackgroundAudio(filename:String, soundVolume:int):void {
			//trace(filename);
			backgroundAudio.playSound(filename, soundVolume);
		}
		
		public function setLanguage(userLanguage:String):void {
			this.userLanguage = userLanguage;
		}
		
		public function setGender(userGender:String):void {
			this.userGender = userGender;
		}
		
		public function setAge(userAge:String):void {
			this.userAge = userAge;
		}
		
		public function setEditMode(b:Boolean):void {
			editMode = b;
		}
		
		public function setKioskID(kioskID:String):void {
			this.kioskID = kioskID;
		}
		
		public function setLoggerURL(loggerURL:String):void {
			this.loggerURL = loggerURL;
			logger.setURL(loggerURL);
		}
		
		public function inactivityTimeout(e:TimerEvent):void {
			trace("=======> KIOSK HAS TIMED OUT DUE TO INACTIVITY");
			main.enableInactivityScreen();
		}
		
		public function inactivityScreenResponse(reset:Boolean):void {
			trace("Inactivity Screen Response: "+reset);
			// if it's true, reset the conductor and all session properties
			// if it's false, reset the inactivity timer
		}
		
		public function startInactivityTimer():void{
			// called by a unit when a video has ended in an interactive module
			trace("===> Inactivity Timer Started");
			inactivityTimer.start();
		}
		
		public function resetInactivityTimer():void{
			// called by a unit when a button has been pressed
			trace("===> Inactivity Timer Reset");
			inactivityTimer.reset();
		}
		
		public function updateSessionID():void{
			// TEMP REPLACEMENT WHILE DEVELOPING LAMP BASED LOGGING
			sessionID = 0;	// going to be stored in Logger object from now on
			logger.getSessionID(Number(kioskID));
			
			// reads and returns the session ID,
			// while iterating it in the file and saving.
			/*
			var file:File = File.desktopDirectory.resolvePath("sessionID.txt");
			if(file.exists){
				var inStream:FileStream = new FileStream();
				inStream.open(file, FileMode.READ);
				sessionID = Number(inStream.readUTFBytes(inStream.bytesAvailable));
				inStream.close();
				trace("SESSION ID: "+ sessionID);
			}
			var outStream = new FileStream();
			outStream.open(file, FileMode.WRITE);
			outStream.writeUTFBytes(String(sessionID+1));
			outStream.close();
			*/
		}
		
		public function print(filename:String):void{
			//trace(File.applicationDirectory.nativePath+"\\"+filename);
			printSocket.connect("127.0.0.1", 10000);
			printFile = filename;
			//printSocket.send(filename);
			//printSocket.close();
		}
		
		public function socketConnectHandler(event:Event):void{
			trace("connected");
			// NEED TO GET ROUTE TO FILE THROUGH SOME OTHER MEANS
			//trace(File.applicationDirectory.nativePath+"\\"+printFile);
			//printSocket.send(File.applicationDirectory.nativePath+"\\"+printFile);
			//printSocket.close();
		}
		
		public function socketIOErrorHandler(event:Event):void{
			trace(event);
		}
		
		public function securityErrorHandler(event:Event):void{
			trace(event);
		}
		
		public function reset():void{
			for(var n:Number=0; n<numChildren; n++){		// for everything on the stage...
				var oldUnit:Unit = getChildAt(n) as Unit;
				if(oldUnit != null){
					oldUnit.reset();	// might call back to reset if not destroyed immediately
					if(contains(oldUnit)){
						removeChild(oldUnit);
					}
				}
			}
			for(var i:Number=0; i<modules.length; i++){
				var mod:Module = modules[i];
				mod = null;		// destroy module completely
			}
			modules = new Object();		// actual module objects addressed by name
			moduleNames = new Array();	// strings of module names
			for(i=0; i<units.length; i++){
				var unit:Unit = units[i];
				unit = null;	// destroy unit completely
			}
			units = new Object();
			unitNames = new Array();
			// RESET LISTENERS HERE
			userLanguage = "";	// defaults
			userGender = "";
			userAge = "";
			getNewSessionID = true;		// get new ID#
			resetInactivityTimer();
			logger.close();
			xmlLoader.load(new URLRequest(xmlFile));
		}
		
		public function resetAndLoadModules(filename:String):void{
			xmlFile = filename;
			reset();
		}
		
		public function start(entryPoint:String):void {
			if(getNewSessionID){
				// fully resetting, so get new session ID# and clear session time
				sessionTimestamp = new Date();
				updateSessionID();
				getNewSessionID = false;
			}
			//logger.start("kiosk"+kioskID+"_session"+sessionID+".txt");	// sets file name and opens file
			
			// START ENTRY UNIT HERE
			var unit:Unit = units[modules[entryPoint].getEntryPoint()];
			unit.addContentListener(unitLoaded);
			unit.loadContent();
			
			//nextModule = modules[entryPoint];
			//nextModule.addContentListener(moduleLoaded);
			//nextModule.loadContent();
		}
		
		public function unitLoaded(unit:Unit, previousUnitName:String):void{
			trace("UNIT "+ unit.getName() + " LOADED");
			if(unit.crossFade){
				unit.alpha = 0;
				TweenLite.to(unit, unit.crossFadeDuration/1000, {alpha:1});
			}
			if(unit.startLogger){
				sessionTimestamp = new Date();
				logger.start("kiosk"+kioskID+"_session"+sessionID+".txt");	// sets file name and opens file
			}
			addChild(unit);
			unit.start();
			currentUnit = unit;								// reference for debugger and logging
			
			/*
			if(previousUnitName != null){					// initial unit will return null
				if(unit.crossFade){							// set up an internal timer to reset the previous unit
					units[previousUnitName].delayedReset(unit.crossFadeDuration);	
				} else {
					units[previousUnitName].reset();		// reset
				}
			}
			*/
			
			
			for(var i:Number=0; i<numChildren-1; i++){		// for everything below the new unit...
				var oldUnit:Unit = getChildAt(i) as Unit;
				if(oldUnit != null){
					if(unit.crossFade){							// set up an internal timer to reset the previous unit
						oldUnit.delayedReset(unit.crossFadeDuration);
					} else {
						oldUnit.reset();
					}
				}
			}
			
			
			stage.focus = stage;
			logUnitEvent(unit.getModule().getName(), unit.getName(), "started");
			updateDebugDisplay(unit.getModule(), unit);
		}
		
		public function updateDebugDisplay(module:Module, unit:Unit):void {
			debugDisplay.updateDisplay(module, unit);
		}
	}
}