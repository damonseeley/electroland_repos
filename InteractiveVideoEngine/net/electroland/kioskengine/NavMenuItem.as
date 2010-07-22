package net.electroland.kioskengine {
	import flash.display.Sprite;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.text.TextFieldAutoSize;
	import flash.display.BlendMode;
	import flash.events.*;
	
	public class NavMenuItem extends Sprite{
		
		private var conductor:Conductor;
		private var unitList:Sprite;
		private var moduleName:String;
		private var textFormat:TextFormat;
		
		/*
		
		NAVMENUITEM.as
		by Aaron Siegel, 9-1-09
		
		Displays a side menu of units when rolled over.
		
		*/
		
		public function NavMenuItem(conductor:Conductor, moduleName:String, textFormat:TextFormat){
			this.conductor = conductor;
			this.moduleName = moduleName;
			this.textFormat = textFormat;
			var itemLabel:TextField = new TextField();
			itemLabel.text = moduleName;
			itemLabel.selectable = false;
			itemLabel.blendMode = BlendMode.LAYER;
			itemLabel.background = true;
			itemLabel.width = 100;
			itemLabel.height = 14;
			//itemLabel.autoSize = TextFieldAutoSize.LEFT;
			itemLabel.setTextFormat(textFormat);
			addChild(itemLabel);
		}
		
		public function buildMenu():void{
			var itemOffset:int = 14;
			unitList = new Sprite();		// container of unit buttons
			var unitNames:Array = conductor.getModule(moduleName).getUnitNames();
			for (var n=0; n<unitNames.length; n++){
				//trace(unitName +" added to nav menu");
				var unitName:String = unitNames[n];
				var unitEntry:Sprite = new Sprite();
				var unitLabel:TextField = new TextField();
				unitLabel.text = unitName;
				unitLabel.selectable = false;
				unitLabel.blendMode = BlendMode.LAYER;
				unitLabel.background = true;
				//unitLabel.autoSize = TextFieldAutoSize.LEFT;
				unitLabel.width = 100;
				unitLabel.height = 14;
				unitLabel.setTextFormat(textFormat);
				unitEntry.y = n*itemOffset;
				unitEntry.buttonMode = true;
				unitEntry.addChild(unitLabel);
				unitEntry.addEventListener(MouseEvent.MOUSE_UP, mousePressed);
				unitList.addChild(unitEntry);
			}
			unitList.visible = false;
			unitList.x = this.width;
			addChild(unitList);
		}
		
		public function showMenu():void{
			unitList.visible = true;
		}
		
		public function hideMenu():void{
			unitList.visible = false;
		}
		
		public function mousePressed(event:MouseEvent):void{
			conductor.jumpToUnit(event.target.text);
		}
		
	}
	
}