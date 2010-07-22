package net.electroland.kioskengine {
	import flash.display.MovieClip;
	import flash.events.*;
	import flash.net.URLLoader;
	import flash.net.URLRequest;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import flash.ui.Mouse;
	import flash.display.StageDisplayState;
	import flash.display.StageScaleMode;
	//import flash.display.NativeWindow;
	//import flash.display.NativeWindowSystemChrome;
	//import flash.desktop.NativeApplication;
	
	public class Main extends MovieClip{
		
		private var unlockStep:int = 0;		// keeps track of which step the code is on
		private var timeOut:int = 30000;	// milliseconds
		//public var inactivityScreen:InactivityScreen;
		public var inactivityScreen:Unit;
		public var debugDisplay:DebugDisplay;
		public var conductor:Conductor;
		//public var adminPanel:AdminPanel;
		public var adminUnit:Unit;
		private var xmlData:XML;
		private var xmlLoader:URLLoader;
		private var unlockTimer:Timer;
		private var mouseHidden:Boolean = false;
		
		/*
		
		MAIN.as
		by Aaron Siegel, 7-14-09
		
		Document Class for the kiosk software.
		
		*/
		
		public function Main(){
			stage.scaleMode = StageScaleMode.NO_SCALE;
			stage.addEventListener(MouseEvent.CLICK, onMouseClickEvent);
			stage.addEventListener(KeyboardEvent.KEY_DOWN, onKeyPressedEvent);
			stage.addEventListener(KeyboardEvent.KEY_UP, onKeyReleasedEvent);
			addEventListener("enterFrame", onEnterFrameEvent);
			//inactivityScreen = new InactivityScreen()	// displays timeout warning
			//inactivityScreen.setSize(stage.stageWidth, stage.stageHeight);
			debugDisplay = new DebugDisplay();			// displays module/unit info
			conductor = new Conductor(this, debugDisplay);	// controls the playback of modules
			conductor.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler)
			//inactivityScreen.addConductor(conductor);
			//debugDisplay.addConductor(conductor);		// moved inside conductor
			addChild(conductor);
			addChild(debugDisplay);						// place above all modules/units
			//adminPanel = new AdminPanel(conductor);
			//addChild(adminPanel);
			unlockTimer = new Timer(10000,1);
			unlockTimer.addEventListener("timer", resetUnlockCode);
			xmlData = new XML();
			xmlLoader = new URLLoader();
			xmlLoader.addEventListener(Event.COMPLETE, parseXML);
			xmlLoader.load(new URLRequest("CONFIG/kiosk_config.xml"));
			
			
			
			//CODE BY DS TO GET STAGE AT 0,0
			//COULD BE REALLY HACKY, INVESTIGATE (only works with AIR)
			//var mainWindow = stage.nativeWindow;
			//mainWindow.x = 0;
			//mainWindow.y = 0;
			//mainWindow.maximize();
		}
		
		public function enableInactivityScreen():void{
			inactivityScreen.loadContent();
			inactivityScreen.addContentListener(inactivityScreenLoaded);
			inactivityScreen.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
			//inactivityScreen.start();
			//addChild(inactivityScreen);
		}
		
		public function disableInactivityScreen():void{
			if(inactivityScreen.running){
				inactivityScreen.reset();
				removeChild(inactivityScreen);
			}
		}
		
		public function inactivityScreenLoaded(unit:Unit, oldUnitName:String):void{
			unit.start();
			addChild(unit);
		}
		
		public function enableAdminPanel():void{
			adminUnit.loadContent();
			adminUnit.addContentListener(adminPanelLoaded);
			adminUnit.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler)
		}
		
		public function disableAdminPanel():void{
			if(adminUnit.running){
				adminUnit.reset();
				removeChild(adminUnit);
			}
		}
		
		public function adminPanelLoaded(unit:Unit, requester:String):void{
			unit.start();
			addChild(unit);
		}
		
		public function itemEventHandler(event:ItemEvent):void {
			// receives events for resetting the conductor or 
			trace("DOCUMENT CLASS ITEM EVENT: "+ event.getAction() +" "+ event.getArguments());
			if(event.getAction() == "wakeup"){
				// reset the inactivity screen and timers
				disableInactivityScreen();
				conductor.resetInactivityTimer();
				conductor.startInactivityTimer();
			} else if(event.getAction() == "resetSession"){
				if(event.getSrc() == "Inactivity_Screen"){
					conductor.logUnitEvent("", event.getSrc(), "SESSION TIMED OUT");
				} else if(event.getSrc() == "Admin_Panel"){
					conductor.logUnitEvent("", event.getSrc(), "SESSION RESET BY ADMIN");
				}
				disableInactivityScreen();
				disableAdminPanel();
				conductor.reset();
			} else if(event.getAction() == "loadModules"){
				conductor.loadModules(event.getArguments());
			} else if(event.getAction() == "resetAndLoadModules"){
				conductor.resetAndLoadModules(event.getArguments());
			} else if(event.getAction() == "exitAdminPanel"){
				disableAdminPanel();
				conductor.resetInactivityTimer();
				conductor.startInactivityTimer();
			} else if(event.getAction() == "volumeUp"){
				conductor.volumeUp();
			} else if(event.getAction() == "volumeDown"){
				conductor.volumeDown();
			} else if(event.getAction() == "quit"){
				//NativeApplication.nativeApplication.exit();
			}
		}
		
		public function onEnterFrameEvent(event:Event):void{
			var now:Date = new Date();
			debugDisplay.updateCounter();
			//trace(now.valueOf() - lastActivity.valueOf());
		}
		
		public function onKeyPressedEvent(event:KeyboardEvent):void{
			//trace("KEYCODE " + event.keyCode);
			if(event.keyCode == 65){	// D
				enableAdminPanel();
			}
			if(event.keyCode == 68){	// D
				debugDisplay.toggleDisplay();
			}
			if(event.keyCode >= 48 && event.keyCode < 59){
				// only works when modules are named uniformly
				conductor.jumpToModule("Module "+(event.keyCode-48));
			}
			/*
			if(event.keyCode == 39){	// right arrow key
				// jump to the end of whatever primary media file is being played
				var currentUnit:Unit = conductor.getCurrentModule().getCurrentUnit();
				if(currentUnit.getType() == "video"){
					
				} else if(currentUnit.getType() == "audio"){
					
				}
			}
			*/
			if(event.keyCode == 71){	// G
				if(!conductor.isEditMode()){
					conductor.setEditMode(true);
				}
			} else if(event.keyCode == 67){	// C
				if(mouseHidden){
					Mouse.show();
					mouseHidden = false;
				} else {
					Mouse.hide();
					mouseHidden = true;
				}
			} else if(event.keyCode == 81){	// Q
				//NativeApplication.nativeApplication.exit();
			}
		}
		
		public function onKeyReleasedEvent(event:KeyboardEvent):void{
			if(event.keyCode == 71){	// G
				conductor.setEditMode(false);
			}
		}
		
		public function onMouseClickEvent(event:MouseEvent):void{
			if(unlockStep == 0){			// upper left
				if(mouseX < stage.stageWidth/10 && mouseY < stage.stageHeight/10){
					unlockTimer.start();
					trace("unlock timer started");
					unlockStep++;
				}
			} else if(unlockStep == 1){		// upper right
				if(mouseX > stage.stageWidth - (stage.stageWidth/10) && mouseY < stage.stageHeight/10){
					unlockStep++;
				}
			} else if(unlockStep == 2){		// lower left
				if(mouseX < stage.stageWidth/10 && mouseY > stage.stageHeight- (stage.stageHeight/10)){
					unlockStep++;
				}
			} else if(unlockStep == 3){		// lower right
				if(mouseX > stage.stageWidth - (stage.stageWidth/10) && mouseY > stage.stageHeight- (stage.stageHeight/10)){
					unlockStep = 0;
					unlockTimer.reset();
					// turn admin panel on here
					//adminPanel.toggleDisplay();
					enableAdminPanel();
				}
			}
		}
		
		public function parseXML(e:Event):void{
			var xmlData:XML = new XML(e.target.data);
			conductor.load(xmlData.ModuleFile);
			conductor.setButtonSound(xmlData.ButtonSound);
			if(xmlData.ShowDebugDisplay == "true"){
				debugDisplay.toggleDisplay();
			}
			if(xmlData.ShowCursor != "true"){
				Mouse.hide();
				mouseHidden = true;
			}
			conductor.setInactivityTimer(Number(xmlData.InactivityTimeout));
			conductor.setKioskID(xmlData.KioskID);
			conductor.setLoggerURL(xmlData.LoggerURL);
			var inactivityUnitList:XMLList = xmlData.InactivityUnit;
			for each (var inactivityUnit:XML in inactivityUnitList) {	// only done once
				inactivityScreen = new Unit(conductor, inactivityUnit);
				inactivityScreen.ignoreTimeout = true;
				conductor.addInactivityScreen(inactivityScreen);
			}
			var adminUnitList:XMLList = xmlData.AdminUnit;
			for each(var adminUnitElement:XML in adminUnitList){
				adminUnit = new Unit(conductor, adminUnitElement);
			}
			if(xmlData.FullScreen == "true"){
				//var nativeWindow:NativeWindow = new NativeWindow();
				//stage.nativeWindow.
				//stage.nativeWindow.systemChrome = NativeWindowSystemChrome.NONE;
				//stage.displayState = StageDisplayState.FULL_SCREEN_INTERACTIVE;
			}
		}
		
		public function resetUnlockCode(e:TimerEvent):void{
			trace("unlock timer reset");
			unlockStep = 0;
		}
		
		
	}
	
}