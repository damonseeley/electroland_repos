package net.electroland.kioskengine {
	import flash.display.Sprite;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.text.TextFieldAutoSize;
	import flash.display.BlendMode;
	import flash.events.Event;
	
	public class DebugDisplay extends Sprite {
		private var conductor:Conductor;
		private var moduleLabel:TextField;
		private var unitLabel:TextField;
		private var unitCounter:TextField;
		private var textFormat:TextFormat;
		//private var moduleMenu:DropDownMenu;
		//private var unitMenu:DropDownMenu;
		private var navMenu:NavMenu;
		
		private var menuLabel:TextField;
		
		/*
		
		DEBUGDISPLAY.as
		by Aaron Siegel, 7-7-09
		
		Used to display the current module ID and description as well as current unit ID and description.
		
		*/
		
		public function DebugDisplay(){
			moduleLabel = new TextField();
			moduleLabel.selectable = false;
			moduleLabel.blendMode = BlendMode.LAYER;
			moduleLabel.autoSize = TextFieldAutoSize.LEFT;
			moduleLabel.background = true;
			moduleLabel.x = 10;
			moduleLabel.y = 10;
			unitLabel = new TextField();
			unitLabel.selectable = false;
			unitLabel.blendMode = BlendMode.LAYER;
			unitLabel.autoSize = TextFieldAutoSize.LEFT;
			unitLabel.background = true;
			unitLabel.x = 10;
			unitLabel.y = 25;
			textFormat = new TextFormat();
			textFormat.font = "Arial";
			textFormat.size = 10;
			moduleLabel.setTextFormat(textFormat);
			unitLabel.setTextFormat(textFormat);
			
			menuLabel = new TextField();
			menuLabel.selectable = false;
			menuLabel.blendMode = BlendMode.LAYER;
			menuLabel.autoSize = TextFieldAutoSize.LEFT;
			menuLabel.background = true;
			menuLabel.x = 500;
			menuLabel.y = 10;
			menuLabel.text = "Kiosk Nav";
			menuLabel.setTextFormat(textFormat);
			addChild(menuLabel);
			
			addChild(moduleLabel);
			addChild(unitLabel);
			
			unitCounter = new TextField();
			unitCounter.selectable = false;
			unitCounter.blendMode = BlendMode.LAYER;
			unitCounter.autoSize = TextFieldAutoSize.LEFT;
			unitCounter.background = true;
			unitCounter.x = 10;
			unitCounter.y = 40;
			unitCounter.text = "Unit Count: 0";
			unitCounter.setTextFormat(textFormat);
			addChild(unitCounter);
			
//			moduleMenu = new DropDownMenu("MODULES");
//			moduleMenu.x = 555;
//			moduleMenu.y = 10;
//			unitMenu = new DropDownMenu("UNITS");
//			unitMenu.x = 618;
//			unitMenu.y = 10;
//			addChild(moduleMenu);
//			addChild(unitMenu);
			navMenu = new NavMenu();
			navMenu.x = 555;
			navMenu.y = 10;
			addChild(navMenu);
			
			this.visible = false;
		}
		
		public function addConductor(conductor:Conductor){
			this.conductor = conductor;
			//moduleMenu.addConductor(conductor);
			//unitMenu.addConductor(conductor);
			navMenu.buildMenu(conductor);
		}
		
		public function updateDisplay(currentModule:Module, currentUnit:Unit):void{
			moduleLabel.text = "MODULE: "+ currentModule.getName() +": "+ currentModule.getDescription();
			unitLabel.text = "UNIT: "+ currentUnit.getName() +": "+ currentUnit.getDescription();
			moduleLabel.setTextFormat(textFormat);
			unitLabel.setTextFormat(textFormat);
			//moduleMenu.updateItems(conductor.getModuleNames());
			//unitMenu.updateItems(conductor.getUnitNames());
		}
		
		public function toggleDisplay(){
			this.visible = !this.visible;
		}
		
		public function updateCounter():void{
			//trace("update counter!");
			if(unitCounter != null && conductor != null){
				unitCounter.text = "Unit Count: "+conductor.numChildren;
				unitCounter.setTextFormat(textFormat);			
			}
		}
		
	}
	
}