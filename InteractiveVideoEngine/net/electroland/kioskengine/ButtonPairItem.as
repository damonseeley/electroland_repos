package net.electroland.kioskengine  {
	
	public class ButtonPairItem extends Item {
		public var buttonList:Array = new Array();
		private var activated:Boolean = false;
		private var buttonState:Boolean = false;
		private var itemsLoaded = 0;
		
		public function ButtonPairItem(conductor:Conductor, itemElement:XML){
			super(conductor);
			parseXML(itemElement);
		}
		
		public function itemEventHandler(event:ItemEvent):void{
			// log the option picked and add to the answers array
			//trace(event.getAction()+" "+event.getArguments());
			activated = true;
			if(event.getArguments() == "true"){
				buttonState = true;
				buttonList[0].select();
				buttonList[1].deselect();
			} else {
				buttonState = false;
				buttonList[0].deselect();
				buttonList[1].select();
			}
			dispatchEvent(new ItemEvent(event.getAction(), event.getArguments()));
		}
		
		public function buttonLoaded(item:Item):void{
			trace(getName()+".itemLoaded(): "+ item.getName() +" LOADED");
			itemsLoaded++;
			if(contentLoadedCallback != null){			// make sure there is a listener established
				if(itemsLoaded == buttonList.length){	// inform the button list this pair has been fully loaded	
					contentLoadedCallback(this);
				}
			}
		}
		
		public function isActivated():Boolean{
			return activated;
		}
		
		public function getState():Boolean{
			return buttonState;
		}
		
		public function loadContent():void{
			for each (var button in buttonList){
				button.addContentListener(buttonLoaded);
				button.loadContent();
				button.deselect();
			}
		}
		
		override protected function parseXML(itemElement:XML):void{
			setID(itemElement.attribute("ID"));
			setName(itemElement.attribute("Name"));
			setType(itemElement.attribute("Type"));
			setDescription(itemElement.attribute("Description"));
			setLayer(Number(itemElement.attribute("Layer")));
			x = Number(itemElement.attribute("x"));				// x/y/width/height never used here
			y = Number(itemElement.attribute("y"));
			setWidth(Number(itemElement.attribute("width")));
			setHeight(Number(itemElement.attribute("height")));
			for each (var itemEventElement:XML in itemElement.ItemEvent){
				//trace(itemEventElement.attribute("Action"));
				parseEffects(itemEventElement);
			}
			for each (var nestedItemElement:XML in itemElement.Item){
				//trace(nestedItemElement);
				var button:ButtonItem = new ButtonItem(conductor, nestedItemElement);
				buttonList.push(button);
			}
			for each (var itemEntryElement:XML in itemElement.ItemEntry){
				parseItemEntries(itemEntryElement);
			}
		}
		
		override public function reset():void{
			// buttons have independent timers
			for each (var button in buttonList){
				removeChild(button);
				button.reset();
				button.removeEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
			}
			super.reset();
		}
		
		public function startTimer():void{
			for each (var button in buttonList){
				addChild(button);
				button.startTimer();
				button.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
			}
			startEffectTimers();
		}
		
	}
	
}