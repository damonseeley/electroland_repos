package net.electroland.kioskengine  {
	import flash.display.Sprite;
	import flash.display.MovieClip;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.text.TextFormatAlign;
	import flash.text.TextFieldAutoSize;
	import flash.display.BlendMode;
	import flash.events.*;
	import flash.display.Loader;
	import flash.net.URLRequest;
	import flash.system.ApplicationDomain;
	import flash.utils.Timer;
	
	public class ButtonItem extends Item{
		private var square:Sprite;
		private var textLabel:TextField;
		private var drawn:Boolean = false;
		private var buttonLoader:Loader
		private var button:MovieClip;			// used for external buttons
		private var buttonLibrary:ApplicationDomain;
		private var buttonLibraryLoaded:Boolean = false;
		private var clickDelayValue:Number = 3000;
		
		/*
		
		BUTTONITEM.as
		by Aaron Siegel, 7-3-09
		
		*/
		
		public function ButtonItem(conductor:Conductor, itemElement:XML){
			super(conductor);
			interactive = true;
			parseXML(itemElement);
		}
		
		override public function effectEventHandler(event:EffectEvent):void{
			if(event.getType() == "activate"){
				if(button != null){
					button.activate();
					trace("button "+getName()+" activate event");
					deactivated = false;
				}
			} else if(event.getType() == "deactivate"){
				if(button != null){
					button.deactivate();
					trace("button "+getName()+" deactivate event");
					deactivated = true;
				}
			} 
			super.effectEventHandler(event);
		}
		
		public function loadContent():void{
			if(!drawn){	// if it hasn't been drawn yet...
				if(getButtonType() == null || getButtonType() == "default"){
					simpleRoundedShape();
					textLabel = new TextField();
					var textFormat:TextFormat = new TextFormat();
					textFormat.font = "Arial";
					textFormat.align = TextFormatAlign.CENTER;
					textFormat.size = 16;
					textFormat.bold = true;
					textLabel.selectable = false;
					textLabel.wordWrap = true;
					textLabel.blendMode = BlendMode.LAYER;
					textLabel.y = getHeight()/2 - getHeight()/10;
					if(itemEntries.hasOwnProperty("all")){				// set text value and potentially override the placement
						textLabel.text = itemEntries["all"].value;
						if(itemEntries["all"].hasOwnProperty("x")){
							x = itemEntries["all"]["x"];
						}
						if(itemEntries["all"].hasOwnProperty("y")){
							y = itemEntries["all"]["y"];
						}
					} else {
						textLabel.text = itemEntries[conductor.getLanguage()].value;
						if(itemEntries[conductor.getLanguage()].hasOwnProperty("x")){
							x = itemEntries[conductor.getLanguage()]["x"];
						}
						if(itemEntries[conductor.getLanguage()].hasOwnProperty("y")){
							y = itemEntries[conductor.getLanguage()]["y"];
						}
					}
					textLabel.autoSize = TextFieldAutoSize.CENTER;
					textLabel.setTextFormat(textFormat);
					addChild(textLabel);
					if(contentLoadedCallback != null){
						contentLoadedCallback(this);	// tell unit this has been loaded
					}
				} else if (getButtonType() == "Invisible"){
					invisibleButton();
					if(contentLoadedCallback != null){
						contentLoadedCallback(this);	// tell unit this has been loaded
					}
				} else if (getButtonType() == "InvisibleDev"){
					invisibleButtonDev();
					if(contentLoadedCallback != null){
						contentLoadedCallback(this);	// tell unit this has been loaded
					}
				} else {
					// use button from buttons.swf
					//square = conductor.getExternalButton(getButtonType());
					//square.width = getWidth();
					//square.height = getHeight();
					//addChild(square);
					trace(getName()+".loadContent(): EXTERNAL BUTTON");
					externalButton = true;	// super class boolean
					buttonLoader = new Loader();	// loads external buttons
					trace(getName()+".loadContent(): BUTTON LOADER INSTANTIATED");
					buttonLoader.contentLoaderInfo.addEventListener(Event.COMPLETE, buttonsLoaded);
					trace(getName()+".loadContent(): BUTTON LISTENER ADDED");
					buttonLoader.load(new URLRequest("MEDIA/SWF/"+getButtonType()+".swf"));		// button type is swf name
					trace(getName()+".loadContent(): BUTTON LOAD CALL MADE");
				}
				//drawn = true;	// only uncomment this to prevent a second load
			} else {
				if(contentLoadedCallback != null){
					contentLoadedCallback(this);	// tell unit this has been loaded
				}
				if(externalButton){
					// check if this button is set for SingleUse
					if(isSingleUse()){
						if(shouldActivate()){
							button.activate();
							deactivated = false;
							trace("BUTTON ACTIVATED");
						} else {
							button.deactivate();
							deactivated = true;
							trace("BUTTON DEACTIVATED");
						}
					}
				}
			}
			
		}
		
		private function shouldActivate():Boolean{
			return deactivated;
			/*
			if(isSingleUse()){
				// check if the module or unit this references has been seen or not
				if(itemEntries.hasOwnProperty("all")){
					if(itemEntries["all"].action == "module"){
						if(conductor.checkModuleState(itemEntries["all"].link)){					// set button to disabled
							//trace("disable button "+getName());
							return false;
						}
					} else if(itemEntries["all"].action == "unit") {
						if(conductor.checkUnitState(itemEntries["all"].link)){						// set button to disabled
							//trace("disable button "+getName());
							return false;
						}
					}
				} else {
					//trace("user language");
					if(itemEntries[conductor.getLanguage()].action == "module"){
						if(conductor.checkModuleState(itemEntries[conductor.getLanguage()].link)){	// set button to disabled
							//trace("disable button "+getName());
							return false;
						}
					} else if(itemEntries[conductor.getLanguage()].action == "unit") {
						if(conductor.checkUnitState(itemEntries[conductor.getLanguage()].link)){	// set button to disabled
							//trace("disable button "+getName());
							return false;
						}
					}
				}
			}
			return true;
			*/
		}
		
		private function buttonsLoaded(e:Event):void{
			trace(getName()+".buttonsLoaded(): EVENT");
			buttonLibrary = e.target.applicationDomain;
			var textValue:String = "";
			if(itemEntries.hasOwnProperty("all")){
				textValue = itemEntries["all"].value;
				if(itemEntries["all"].hasOwnProperty("x")){
					x = itemEntries["all"].hasOwnProperty("x");
				}
				if(itemEntries["all"].hasOwnProperty("y")){
					y = itemEntries["all"].hasOwnProperty("y");
				}
			} else {
				textValue = itemEntries[conductor.getLanguage()].value;
				if(itemEntries[conductor.getLanguage()].hasOwnProperty("x")){
					x = itemEntries[conductor.getLanguage()]["x"];
					trace("item entry X: "+ itemEntries[conductor.getLanguage()]["x"] +" real X: "+x);
				}
				if(itemEntries[conductor.getLanguage()].hasOwnProperty("y")){
					y = itemEntries[conductor.getLanguage()]["y"];
				}
			}
			var ExternalButton:Class = buttonLibrary.getDefinition("net.electroland.buttons."+getButtonType()) as Class;
			
			// ds add the button MinWidth code here
			var minWidth:Number = 0;
			if(itemEntries.hasOwnProperty("all") && itemEntries["all"].hasOwnProperty("minWidth")){
				minWidth = itemEntries["all"].minWidth;
			} else if (itemEntries[conductor.getLanguage()].hasOwnProperty("minWidth")) {
				minWidth = itemEntries[conductor.getLanguage()].minWidth;
			}
			trace("MINWIDTH = ", minWidth);
			
			button = new ExternalButton(externalButtonCallback, textValue, minWidth) as MovieClip;	// provide callback and visible text value
			
			// check if this is a singleUse button, and if the button should be deactivated
			if(isSingleUse()){
				if(!deactivated){
					button.activate();
					deactivated = false;
					trace("BUTTON ACTIVATED");
				} else {
					button.deactivate();
					deactivated = true;
					trace("BUTTON DEACTIVATED");
				}
			}
			//button.activate();
			//deactivated = false;
			//added by DS to get hand cursor
			button.buttonMode = true;
			trace("BUTTON ACTIVATED");
				
			if(getHorizontalAlign() == "center"){
				button.x = 0 - button.width/2;
			} else if(getHorizontalAlign() == "right"){
				button.x = 0 - button.width;
			}
			if(getVerticalAlign() == "center"){
				button.y = 0 - button.height/2;
			} else if(getVerticalAlign() == "bottom"){
				button.y = 0 - button.height;
			}
			addChild(button);
			buttonLibraryLoaded = true;
			if(contentLoadedCallback != null){
				contentLoadedCallback(this);	// tell unit this has been loaded
			}
		}
		
		private function externalButtonCallback(b:Object, s:String):void{
			trace("external button pressed");
			if(!conductor.isEditMode()){
				if(!pressed){
					if(itemEntries.hasOwnProperty("all")){
						dispatchEvent(new ItemEvent(itemEntries["all"].action, itemEntries["all"].link));
					} else {
						dispatchEvent(new ItemEvent(itemEntries[conductor.getLanguage()].action, itemEntries[conductor.getLanguage()].link));
					}
					conductor.logItemEvent(" ", " ", getName(), getDescription(), "button pressed");
					
					pressed = true;
					if(isSingleUse() && !deactivated){
						deactivated = true;
					}
					clickDelay = new Timer(clickDelayValue,1);
					clickDelay.addEventListener("timer", clickDelayListener);
					clickDelay.start();
				}
			}
		}
		
		private function invisibleButton():void{
			square = new Sprite();
			addChild(square);
			square.graphics.lineStyle(3,0xaaaaaa);
			square.graphics.beginFill(0xcccccc, 1.0);
			square.graphics.drawRect(0,0,getWidth(),getHeight());
			square.graphics.endFill();
			square.alpha = 0;
			//added by DS to get hand cursor
			square.buttonMode = true;
			
		}
		
		private function invisibleButtonDev():void{
			square = new Sprite();
			addChild(square);
			square.graphics.lineStyle(3,0xaaaaaa);
			square.graphics.beginFill(0xcccccc, 1.0);
			square.graphics.drawRect(0,0,getWidth(),getHeight());
			square.graphics.endFill();
			square.alpha = 0.25;
			//added by DS to get hand cursor
			square.buttonMode = true;
			
		}
		
		private function simpleRoundedShape():void{
			square = new Sprite();
			addChild(square);
			square.graphics.lineStyle(3,0xaaaaaa);
			square.graphics.beginFill(0xcccccc, 0.8);
			square.graphics.drawRoundRect(0,0,getWidth(),getHeight(),20);
			square.graphics.endFill();
			//trace("square drawn");
		}
		
		public function startTimer():void{
			// begin timers on various effects
			startEffectTimers();
		}
		
		public function select():void{
			if(button != null){
				button.select();
			}
		}
		
		public function deselect():void{
			if(button != null){
				button.deselect();
			}
		}
		
		override public function reset():void{
			if(square != null){
				removeChild(square);
			}
			if(button != null){
				removeChild(button);
			}
			buttonLibraryLoaded = false;
			super.reset();
		}
		
	}
	
}