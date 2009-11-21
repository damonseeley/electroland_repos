import controlP5.*;
import processing.opengl.*;
import java.util.concurrent.ConcurrentHashMap;
import mpe.client.*;

// FIL TEXT CLOUD
// by Electroland
//
// Java/Processing development by Aaron Siegel, October/November 2009.
//
// Projected Area: 100'2" x 18'8" = 3072 x 768
// Interface Area: 40" diagonal   = 1920 x 1080
// Ramp Area:      54'3" x 6'5" ramped down to 2'7"



PGraphicsOpenGL pgl;             // opengl stuff
ControlP5 controlP5;             // controls for debugging
TCPClient client;                // used for MPE communication

// DATA STORAGE VARIABLES
Properties properties            = new Properties();
ConcurrentHashMap textBlocks     = new ConcurrentHashMap();
ConcurrentHashMap authorObjects  = new ConcurrentHashMap();
ConcurrentHashMap quoteObjects   = new ConcurrentHashMap();
ConcurrentHashMap bioObjects     = new ConcurrentHashMap();
ArrayList genreList_english      = new ArrayList();
ArrayList genreList_spanish      = new ArrayList();
int numAuthors                   = 0;
String xmlFileName;

// APPLET DIMENSION VARIABLES
int xSize, ySize;                // stage width/height
float xGrav, yGrav;              // location of gravity center
float xgravity, ygravity;        // gravitational force
float interfaceScale = 1.6;      // scaling of the text cloud for master control machine
float defaultInterfaceScale;
int verticalOffset = 0;
int horizontalOffset = 0;        // necessary for slider widget to operate
boolean yflip = false;
boolean standAlone = false;      // for testing without MPE

// TEXT VARIABLES
String authorFontName, quoteFontName, bioFontName;  // font name
String dateFontName, genreFontName;
int authorFontSize, quoteFontSize, bioFontSize;     // font size
int dateFontSize, genreFontSize;
float textScaleMin, textScaleMax;                   // scaling of textblocks (0-1 being 100%)
float activatedScaleMax;                            // the scale author names inflate to when activated
float textMarginVertical, textMarginHorizontal;     // margin around text blocks
float quoteMarginHorizontal, quoteMarginVertical;   // specific buffer just for quotes/bios
int overflow;                                       // amount of text overflow on each side of stage before wrapping
int numTextBlocks, authorNum;                       // keeping track of textblock creation since application start
float textVerticalDamping, textHorizontalDamping;   // default motion damping for all text
float textHorizontalSpring, textVerticalSpring;     // multiplier for moving textblocks away from each other
int textRollOverDuration, textRollOutDuration;      // author name drag-over effect
float textRollOverScale, textRollOverAlpha;         // scale and opacity multipliers when dragged over
float quoteTextScale;                               // scale for quote text
float bioTextScale;
float dateTextScale;
float genreTextScale;
int quoteIntroDelay;                                // delay to wait for textblocks to get pushed out before revealing quote
int quoteFadeInDuration;
int quoteHoldDuration;
int quoteFadeOutDuration;
int quoteDistanceThreshold;
int quotationOffset;
int authorFadeOutDelay;    // additional delay author name waits AFTER quote/bio has disappeared from stage
int charsPerLine;
int quoteID = 0;
int maxQuotes;
float clearAreaMultiplier;
float quotePushMultiplier;
int quoteBlockTopMargin, bioBlockTopMargin, dateTextLeftMargin, genreTextTopMargin;

// COLOR VARIABLES
int backgroundGray;
int authorTextRedVal, authorTextGreenVal, authorTextBlueVal;
int authorSelectedRedVal, authorSelectedGreenVal, authorSelectedBlueVal;
int quoteTextRedVal, quoteTextGreenVal, quoteTextBlueVal;
int bioTextRedVal, bioTextGreenVal, bioTextBlueVal;
int dateTextRedVal, dateTextGreenVal, dateTextBlueVal;
int genreTextRedVal, genreTextGreenVal, genreTextBlueVal;
int textAlphaMax, textAlphaFallOff;
int horizontalFallOff;

// CONTROL VARIABLES
float dragDamp;                  // slows users dragging action
float dragRadius;
float postDragDelay;             // prevents accidental release
boolean addAuthor = false;
boolean displayStats = false;
boolean displayControls = false;
boolean displayFrames = false;
boolean displayBoundingBoxes = false;
boolean enableCamera = false;
boolean zooming = false;
boolean displayCursor = false;
int zoomCounter, zoomDuration, zoomDelayCounter;
int buttonZoomDuration, inactivityZoomDuration;
float zoomTarget, zoomStart;
float inactivityZoomDelay;
float maxZoom, minZoom;
PFont statFont;
long lastDragged;
int cameraX, cameraY, cameraZ;
WidgetManager widgetManager;
Button btnEnglish, btnEspanol, btnAuthorCloud, btnDate, btnPlus, btnMinus, btnPopularity;
DropDownMenu dropdownGenre;
Balloon balloon;
Slider slider;
float horizontalMouseOffset, verticalMouseOffset;
float scaledWidth, scaledHeight;

// BUTTON VARIABLES
PImage buttonAuthorCloudImage, buttonDateImage, buttonGenreImage, buttonPopularityImage;
PImage buttonAuthorCloudDown, buttonDateDown, buttonGenreDown, buttonPopularityDown;
int buttonAuthorCloudX, buttonAuthorCloudY, buttonDateX, buttonDateY;
int buttonGenreX, buttonGenreY, buttonPopularityX, buttonPopularityY;
PImage buttonEnglishImage, buttonEspanolImage, buttonPlusImage, buttonMinusImage, buttonSliderImage;
PImage buttonEnglishDown, buttonEspanolDown, buttonPlusDown, buttonMinusDown;
int buttonEnglishX, buttonEnglishY, buttonEspanolX, buttonEspanolY;
int buttonPlusX, buttonPlusY, buttonMinusX, buttonMinusY;
int buttonSliderX, buttonSliderY, dropDownItemHeight, dropDownItemLeading;
PImage buttonQuoteImage, buttonQuoteDown, buttonBiographyImage, buttonBiographyDown, balloonImage, balloonDown;
PImage buttonQuoteEspImage, buttonQuoteEspDown, buttonBiographyEspImage, buttonBiographyEspDown, buttonQuoteGrey, buttonBiographyGrey;
PImage backgroundImage, leftFade, rightFade;
PImage sliderBarSegmentImage, sliderBarSegmentDown, sliderBarLeftImage, sliderBarLeftDown, sliderBarRightImage, sliderBarRightDown;

// IMAGE VARIABLES
PImage imageLanguageInstructionsImage, imageViewingInstructionsImage, imageZoomTitleImage, imageTitleImage;
int imageLanguageInstructionsX, imageLanguageInstructionsY, imageViewingInstructionsX, imageViewingInstructionsY;
int imageZoomTitleX, imageZoomTitleY, imageTitleX, imageTitleY;

// USER VARIABLES
String userLanguage = "English"; // "Espanol";
String mode = "authorcloud";     // "authorcloud", "date", "genre", "popularity"
String currentGenre;             // name of genre being displayed

// SCREENSAVER VARIABLES
int screensaverInactivity;              // number of frames until screensaver is turned on
float screensaverSpeed;                 // velocity applied to each text block
boolean screensaverActivated = false;   // boolean to kick on application of velocity
int inactivityCounter = 0;              // keep track of last user action
int resetDelay;                         // # of frames to wait after screensaver is activated to reset everything
int resetCounter = 0;                   // counter to reset
int resetDuration;

// STAT VARIABLES
long lastTime;
int mpeFps = 0;
ArrayList mpeFpsHistory = new ArrayList();

// RAMP MASK VARIABLES
int rampMaskTopLeftX, rampMaskTopLeftY;
int rampMaskTopRightX, rampMaskTopRightY;
int rampMaskBottomRightX, rampMaskBottomRightY;
int rampMaskBottomLeftX, rampMaskBottomLeftY;






// INIT FUNCTIONS

void setup(){    
  loadProperties();
  
  client = new TCPClient(sketchPath("mpe.ini"), this);
  size(client.getLWidth(), client.getLHeight(), OPENGL);
  randomSeed(1);
  if(!displayCursor){
    noCursor();
  }
  xGrav = client.getMWidth()/2;
  yGrav = client.getMHeight()/2;
  colorMode(RGB, 255);
  pgl = (PGraphicsOpenGL)g;
  statFont = loadFont("ArialMT-10.vlw");
  
  if(displayControls){
    loadControls();
  }
  
  // TEMPORARY! MUST GET GENRE LIST FROM BILINGUAL FILE
  genreList_english.add("fiction");
  genreList_english.add("nonfiction");
  
  loadXML(xmlFileName);
  if(!standAlone){
    client.start();
  }
  lastTime = System.currentTimeMillis();  // record time for FPS comparison
}

public void loadXML(String filename){
  XMLElement xml = new XMLElement(this, sketchPath(filename));
  numAuthors = xml.getChildCount();
  for(int i=0; i<xml.getChildCount(); i++){
    XMLElement authorXML = xml.getChild(i);
    
    String name = "";
    String seminalwork_english = "";
    String seminalwork_spanish = "";
    String bio_english = "";
    String bio_spanish = "";
    String quote_english = "";
    String quote_spanish = "";
    String[] genres = new String[0];
    String[] geographies = new String[0];
    int born = 0;
    int died = 0;
    int workbegan = 0;
    int workended = 0;
      
    for(int n=0; n<authorXML.getChildCount(); n++){
      XMLElement entry = authorXML.getChild(n);
      //println(authorXML.getChild(n).getChildCount());
      
      if(entry.getContent() != null || entry.getChildCount() > 0){
        if("Name".equals(entry.getName())){
          name = entry.getContent();
        } else if("Born".equals(entry.getName())){
          if(entry.getContent().endsWith("s")){
            born = Integer.parseInt(entry.getContent().substring(0,4));
          } else if ("present".equals(entry.getContent())){
            born = 2009;
          } else {
            born = Integer.parseInt(entry.getContent());
          }
        } else if("Died".equals(entry.getName())){
          if(entry.getContent().endsWith("s")){
            died = Integer.parseInt(entry.getContent().substring(0,4));
          } else {
            died = Integer.parseInt(entry.getContent());
          }
        } else if("WorkBegan".equals(entry.getName())){
          if(entry.getContent().endsWith("s")){
            workbegan = Integer.parseInt(entry.getContent().substring(0,4));
          } else {
            workbegan = Integer.parseInt(entry.getContent());
          }
        } else if("WorkEnded".equals(entry.getName())){
          if(entry.getContent().endsWith("s")){
            workended = Integer.parseInt(entry.getContent().substring(0,4));
          } else if ("present".equals(entry.getContent())){
            // work has not ended, so leave it null
          } else {
            workended = Integer.parseInt(entry.getContent());
          }
        } else if("SeminalWork".equals(entry.getName())){
          // TODO: switch to english/spanish
          //seminalwork_english = entry.getContent();
          for(int q=0; q<entry.getChildCount(); q++){
            if(entry.getChild(q).getContent() != null){
              if("English".equals(entry.getChild(q).getName())){
                seminalwork_english = entry.getChild(q).getContent();
              } else if("Spanish".equals(entry.getChild(q).getName())){
                seminalwork_spanish = entry.getChild(q).getContent();
              }
            }
          }
        } else if("AuthorDescription".equals(entry.getName())){
          // TODO: switch to Biography and english/spanish
          //bio_english = entry.getContent();
          for(int q=0; q<entry.getChildCount(); q++){
            if(entry.getChild(q).getContent() != null){
              if("English".equals(entry.getChild(q).getName())){
                bio_english = entry.getChild(q).getContent();
              } else if("Spanish".equals(entry.getChild(q).getName())){
                bio_spanish = entry.getChild(q).getContent();
              }
            }
          }
        } else if("Geographies".equals(entry.getName())){
          geographies = trim(entry.getContent().split(","));
        } else if("Genres".equals(entry.getName())){
          genres = trim(entry.getContent().split(","));
        } else if("Quote".equals(entry.getName())){
          for(int q=0; q<entry.getChildCount(); q++){
            if(entry.getChild(q).getContent() != null){
              if("English".equals(entry.getChild(q).getName())){
                quote_english = entry.getChild(q).getContent();
                //if(quote_english.split("\\*").length > 1){
                  //println(quote_english);
                //}
              } else if("Spanish".equals(entry.getChild(q).getName())){
                quote_spanish = entry.getChild(q).getContent();
              }
            }
          }
        }
      }
      
    }
    // CREATE AUTHOR HERE
    createAuthor(i, name, born, died, workbegan, workended, seminalwork_english, seminalwork_spanish, geographies, genres, quote_english, quote_spanish, bio_english, bio_spanish);
  }
}

void createAuthor(int id, String name, int born, int died, int workbegan, int workended,
                  String seminalwork_english, String seminalwork_spanish, String[] geographies,
                  String[] genres, String quote_english, String quote_spanish, String bio_english,
                  String bio_spanish){
                    
  float xpos = ((id % 10) * ((client.getMWidth()+(overflow*2)) / 10)) - overflow;
  float ypos;
  if((id/10) % 2 > 0){
    ypos = (client.getMHeight()/2) - ((id / 10) * 10);
  } else {
    ypos = (client.getMHeight()/2) + ((id / 10) * 10);
  }
  // ORIGINAL SCALING ALGORITHM
  float textscale = random(textScaleMin, textScaleMin + (((100-id) * 0.01) * (textScaleMax-textScaleMin)));
  // UNUSED NEW SCALING ALGORITHM
  //float scaleRange = (((numAuthors-id) * (1.0/numAuthors)) * (textScaleMax-textScaleMin));
  //println(scaleRange);
  //float textscale = random(textScaleMin, textScaleMin + scaleRange);
  
  Author author = new Author(id, name, quote_english, xpos, ypos, authorFontName, authorFontSize, textscale);
  author.setRed(authorTextRedVal);
  author.setGreen(authorTextGreenVal);
  author.setBlue(authorTextBlueVal);
  author.setAlphaMax(textAlphaMax);
  author.setAlphaFallOff(textAlphaFallOff);
  author.setXMargin(textMarginHorizontal);
  author.setYMargin(textMarginVertical);
  author.setHorizontalDamping(textHorizontalDamping);
  author.setVerticalDamping(textVerticalDamping);
  author.setHorizontalSpring(textHorizontalSpring);
  author.setVerticalSpring(textVerticalSpring);
  author.setRollOverDuration(textRollOverDuration);
  author.setRollOutDuration(textRollOutDuration);
  author.setRollOverScale(textRollOverScale);
  author.setRollOverAlpha(textRollOverAlpha);
  author.setSelectedRed(authorSelectedRedVal);
  author.setSelectedGreen(authorSelectedGreenVal);
  author.setSelectedBlue(authorSelectedBlueVal);
  author.setStageWidth(client.getMWidth());
  author.setStageHeight(client.getMHeight());
  author.setMaxScale(activatedScaleMax);
  author.softenCollisions();  // tweens damping
  
  author.born = born;
  author.died = died;
  author.workbegan = workbegan;
  author.workended = workended;
  author.seminalwork_english = seminalwork_english;
  author.seminalwork_spanish = seminalwork_spanish;
  author.geographies = geographies;
  author.genres = genres;
  author.quote_english = quote_english;
  author.quote_spanish = quote_spanish;
  author.bio_english = bio_english;
  author.bio_spanish = bio_spanish;
  
  textBlocks.put(id, author);
  authorObjects.put(id, author);
  numTextBlocks++;
  authorNum++;
}

void loadControls(){  
  widgetManager   = new WidgetManager(client);
  btnEnglish      = new Button("English", buttonEnglishX, buttonEnglishY, 1, buttonEnglishImage, buttonEnglishDown);
  btnEnglish.silentOn();      // defaults to on
  btnEspanol      = new Button("Espanol", buttonEspanolX, buttonEspanolY, 0, buttonEspanolImage, buttonEspanolDown);
  //btnAuthorCloud  = new Button("Author Cloud", buttonAuthorCloudX, buttonAuthorCloudY, 2, buttonAuthorCloudImage, buttonAuthorCloudDown);
  //btnAuthorCloud.silentOn();  // defaults to on
  //btnDate         = new Button("By Date", buttonDateX, buttonDateY, 3, buttonDateImage, buttonDateDown);
  //btnPopularity   = new Button("By Popularity", buttonPopularityX, buttonPopularityY, 4, buttonPopularityImage, buttonPopularityDown);
  //dropdownGenre   = new DropDownMenu("By Genre", buttonGenreX, buttonGenreY, 5, dropDownItemHeight, dropDownItemLeading, buttonGenreImage, buttonGenreDown);
  //dropdownGenre.addItem("Fiction", 0);  // these will be added automatically from a props file
  //dropdownGenre.addItem("Non-Fiction", 1);
  btnPlus         = new Button("ZoomIn", buttonPlusX, buttonPlusY, 6, buttonPlusImage, buttonPlusDown);
  btnMinus        = new Button("ZoomOut", buttonMinusX, buttonMinusY, 7, buttonMinusImage, buttonMinusDown);
  slider          = new Slider("Slider", buttonSliderX, buttonSliderY, 8, buttonSliderImage, sliderBarSegmentImage, sliderBarSegmentDown, sliderBarLeftImage, sliderBarLeftDown, sliderBarRightImage, sliderBarRightDown);
  scaledWidth = client.getMWidth() * (1/interfaceScale);
  scaledHeight = client.getMHeight() * (1/interfaceScale);
  horizontalMouseOffset = (scaledWidth/2) - (width/2 + horizontalOffset);      // centered
  verticalMouseOffset = (scaledHeight/2) - (height/2 + verticalOffset);         // centered
  slider.setAreaVisible(width/(float)scaledWidth);
  slider.setOffset(horizontalMouseOffset/(float)scaledWidth);
  
  widgetManager.addItem(btnEnglish);
  widgetManager.addItem(btnEspanol);
  //widgetManager.addItem(btnAuthorCloud);
  //widgetManager.addItem(btnDate);
  //widgetManager.addItem(btnPopularity);
  //widgetManager.addItem(dropdownGenre);
  widgetManager.addItem(btnPlus);
  widgetManager.addItem(btnMinus);
  widgetManager.addItem(slider);
}

void loadProperties(){
  try{
    //InputStream in = createInput("properties.txt");  // load in the properties for this project
    InputStream in = createInput(sketchPath("properties.txt"));
    properties.load(in);
    in.close();
  } catch(Exception e){
    e.printStackTrace();
    System.exit(0);
  }

  xmlFileName             = properties.getProperty("xmlFileName");
  overflow                = Integer.parseInt(properties.getProperty("overflow"));  // amount of text overflow on each side of stage before wrapping
  xgravity                = Float.parseFloat(properties.getProperty("xgravity"));  // gravitational force on text blocks
  ygravity                = Float.parseFloat(properties.getProperty("ygravity"));
  yflip                   = Boolean.parseBoolean(properties.getProperty("yflip"));
  interfaceScale          = Float.parseFloat(properties.getProperty("interfaceScale"));
  defaultInterfaceScale   = interfaceScale;
  //verticalOffset        = Integer.parseInt(properties.getProperty("verticalOffset"));
  dragDamp                = Float.parseFloat(properties.getProperty("dragDamp"));  // amount of damping user control
  dragRadius              = Float.parseFloat(properties.getProperty("dragRadius"));
  postDragDelay           = Float.parseFloat(properties.getProperty("postDragDelay"));
  dropDownItemHeight      = Integer.parseInt(properties.getProperty("dropDownItemHeight"));
  dropDownItemLeading     = Integer.parseInt(properties.getProperty("dropDownItemLeading"));
  //standAlone              = Boolean.parseBoolean(properties.getProperty("standAlone"));
  buttonZoomDuration      = Integer.parseInt(properties.getProperty("zoomDuration"));
  maxZoom                 = Float.parseFloat(properties.getProperty("maxZoom"));
  minZoom                 = Float.parseFloat(properties.getProperty("minZoom"));
  
  backgroundGray          = Integer.parseInt(properties.getProperty("backgroundGray"));      // color properties
  authorTextRedVal        = Integer.parseInt(properties.getProperty("authorTextRedVal"));
  authorTextGreenVal      = Integer.parseInt(properties.getProperty("authorTextGreenVal"));
  authorTextBlueVal       = Integer.parseInt(properties.getProperty("authorTextBlueVal"));
  textAlphaMax            = Integer.parseInt(properties.getProperty("textAlphaMax"));
  textAlphaFallOff        = Integer.parseInt(properties.getProperty("textAlphaFallOff"));
  horizontalFallOff       = Integer.parseInt(properties.getProperty("horizontalFallOff"));
  authorSelectedRedVal    = Integer.parseInt(properties.getProperty("authorSelectedRedVal"));
  authorSelectedGreenVal  = Integer.parseInt(properties.getProperty("authorSelectedGreenVal"));
  authorSelectedBlueVal   = Integer.parseInt(properties.getProperty("authorSelectedBlueVal"));
  quoteTextRedVal         = Integer.parseInt(properties.getProperty("quoteTextRedVal"));
  quoteTextGreenVal       = Integer.parseInt(properties.getProperty("quoteTextGreenVal"));
  quoteTextBlueVal        = Integer.parseInt(properties.getProperty("quoteTextBlueVal"));
  quoteDistanceThreshold  = Integer.parseInt(properties.getProperty("quoteDistanceThreshold"));
  quotationOffset         = Integer.parseInt(properties.getProperty("quotationOffset"));
  
  authorFontName        = properties.getProperty("authorFontName");                        // text properties
  authorFontSize        = Integer.parseInt(properties.getProperty("authorFontSize"));
  quoteFontName         = properties.getProperty("quoteFontName");                        
  quoteFontSize         = Integer.parseInt(properties.getProperty("quoteFontSize"));
  textScaleMin          = Float.parseFloat(properties.getProperty("textScaleMin"));
  textScaleMax          = Float.parseFloat(properties.getProperty("textScaleMax"));
  activatedScaleMax     = Float.parseFloat(properties.getProperty("activatedScaleMax"));
  quoteTextScale        = Float.parseFloat(properties.getProperty("quoteTextScale"));
  textMarginVertical    = Float.parseFloat(properties.getProperty("textMarginVertical"));
  textMarginHorizontal  = Float.parseFloat(properties.getProperty("textMarginHorizontal"));
  quoteMarginHorizontal = Float.parseFloat(properties.getProperty("quoteMarginHorizontal"));
  quoteMarginVertical   = Float.parseFloat(properties.getProperty("quoteMarginVertical"));
  charsPerLine          = Integer.parseInt(properties.getProperty("charsPerLine"));
  maxQuotes             = Integer.parseInt(properties.getProperty("maxQuotes"));
  textVerticalDamping   = Float.parseFloat(properties.getProperty("textVerticalDamping"));
  textHorizontalDamping = Float.parseFloat(properties.getProperty("textHorizontalDamping")); 
  textHorizontalSpring  = Float.parseFloat(properties.getProperty("textHorizontalSpring"));// springy action between textblocks
  textVerticalSpring    = Float.parseFloat(properties.getProperty("textVerticalSpring"));
  quotePushMultiplier   = Float.parseFloat(properties.getProperty("quotePushMultiplier"));
  quoteBlockTopMargin   = Integer.parseInt(properties.getProperty("quoteBlockTopMargin"));
  
  bioFontName           = properties.getProperty("bioFontName");
  bioFontSize           = Integer.parseInt(properties.getProperty("bioFontSize"));
  bioTextScale          = Float.parseFloat(properties.getProperty("bioTextScale"));
  bioTextRedVal         = Integer.parseInt(properties.getProperty("bioTextRedVal"));
  bioTextGreenVal       = Integer.parseInt(properties.getProperty("bioTextGreenVal"));
  bioTextBlueVal        = Integer.parseInt(properties.getProperty("bioTextBlueVal"));
  bioBlockTopMargin     = Integer.parseInt(properties.getProperty("bioBlockTopMargin"));
  
  dateFontName          = properties.getProperty("dateFontName");
  dateFontSize          = Integer.parseInt(properties.getProperty("dateFontSize"));
  dateTextScale         = Float.parseFloat(properties.getProperty("dateTextScale"));
  dateTextRedVal        = Integer.parseInt(properties.getProperty("dateTextRedVal"));
  dateTextGreenVal      = Integer.parseInt(properties.getProperty("dateTextGreenVal"));
  dateTextBlueVal       = Integer.parseInt(properties.getProperty("dateTextBlueVal"));
  dateTextLeftMargin    = Integer.parseInt(properties.getProperty("dateTextLeftMargin"));  
  
  genreFontName         = properties.getProperty("genreFontName");
  genreFontSize         = Integer.parseInt(properties.getProperty("genreFontSize"));
  genreTextScale        = Float.parseFloat(properties.getProperty("genreTextScale"));
  genreTextRedVal       = Integer.parseInt(properties.getProperty("genreTextRedVal"));
  genreTextGreenVal     = Integer.parseInt(properties.getProperty("genreTextGreenVal"));
  genreTextBlueVal      = Integer.parseInt(properties.getProperty("genreTextBlueVal"));
  genreTextTopMargin    = Integer.parseInt(properties.getProperty("genreTextTopMargin"));
  
  screensaverInactivity   = Integer.parseInt(properties.getProperty("screensaverInactivity"));
  screensaverSpeed        = Float.parseFloat(properties.getProperty("screensaverSpeed"));
  resetDelay              = Integer.parseInt(properties.getProperty("resetDelay"));
  resetDuration           = Integer.parseInt(properties.getProperty("resetDuration"));
  inactivityZoomDelay     = Integer.parseInt(properties.getProperty("inactivityZoomDelay"));
  inactivityZoomDuration  = Integer.parseInt(properties.getProperty("inactivityZoomDuration"));
  
  textRollOverDuration  = Integer.parseInt(properties.getProperty("textRollOverDuration"));
  textRollOutDuration   = Integer.parseInt(properties.getProperty("textRollOutDuration"));
  textRollOverScale     = Float.parseFloat(properties.getProperty("textRollOverScale"));
  textRollOverAlpha     = Float.parseFloat(properties.getProperty("textRollOverAlpha"));
  quoteIntroDelay       = Integer.parseInt(properties.getProperty("quoteIntroDelay"));
  quoteFadeInDuration   = Integer.parseInt(properties.getProperty("quoteFadeInDuration"));
  quoteHoldDuration     = Integer.parseInt(properties.getProperty("quoteHoldDuration"));  // duration PER line
  quoteFadeOutDuration  = Integer.parseInt(properties.getProperty("quoteFadeOutDuration"));
  authorFadeOutDelay    = Integer.parseInt(properties.getProperty("authorFadeOutDelay"));
  clearAreaMultiplier   = Float.parseFloat(properties.getProperty("clearAreaMultiplier"));
  
  enableCamera          = Boolean.parseBoolean(properties.getProperty("enableCamera"));   // camera properties
  displayControls       = Boolean.parseBoolean(properties.getProperty("displayControls"));
  displayCursor         = Boolean.parseBoolean(properties.getProperty("displayCursor"));
  
  /*
  buttonAuthorCloudImage  = loadImage(properties.getProperty("buttonAuthorCloudImage"));                // button properties
  buttonAuthorCloudDown   = loadImage(properties.getProperty("buttonAuthorCloudDown"));
  buttonAuthorCloudX      = Integer.parseInt(properties.getProperty("buttonAuthorCloudX"));
  buttonAuthorCloudY      = Integer.parseInt(properties.getProperty("buttonAuthorCloudY"));
  buttonDateImage         = loadImage(properties.getProperty("buttonDateImage"));
  buttonDateDown          = loadImage(properties.getProperty("buttonDateDown"));
  buttonDateX             = Integer.parseInt(properties.getProperty("buttonDateX"));
  buttonDateY             = Integer.parseInt(properties.getProperty("buttonDateY"));
  buttonGenreImage        = loadImage(properties.getProperty("buttonGenreImage"));
  buttonGenreDown         = loadImage(properties.getProperty("buttonGenreDown"));
  buttonGenreX            = Integer.parseInt(properties.getProperty("buttonGenreX"));
  buttonGenreY            = Integer.parseInt(properties.getProperty("buttonGenreY"));
  buttonPopularityImage   = loadImage(properties.getProperty("buttonPopularityImage"));
  buttonPopularityDown    = loadImage(properties.getProperty("buttonPopularityDown"));
  buttonPopularityX       = Integer.parseInt(properties.getProperty("buttonPopularityX"));
  buttonPopularityY       = Integer.parseInt(properties.getProperty("buttonPopularityY"));
  */
  buttonEnglishImage      = loadImage(properties.getProperty("buttonEnglishImage"));
  buttonEnglishDown       = loadImage(properties.getProperty("buttonEnglishDown"));
  buttonEnglishX          = Integer.parseInt(properties.getProperty("buttonEnglishX"));
  buttonEnglishY          = Integer.parseInt(properties.getProperty("buttonEnglishY"));
  buttonEspanolImage      = loadImage(properties.getProperty("buttonEspanolImage"));
  buttonEspanolDown       = loadImage(properties.getProperty("buttonEspanolDown"));
  buttonEspanolX          = Integer.parseInt(properties.getProperty("buttonEspanolX"));
  buttonEspanolY          = Integer.parseInt(properties.getProperty("buttonEspanolY"));
  buttonPlusImage         = loadImage(properties.getProperty("buttonPlusImage"));
  buttonPlusDown          = loadImage(properties.getProperty("buttonPlusDown"));
  buttonPlusX             = Integer.parseInt(properties.getProperty("buttonPlusX"));
  buttonPlusY             = Integer.parseInt(properties.getProperty("buttonPlusY"));
  buttonMinusImage        = loadImage(properties.getProperty("buttonMinusImage"));
  buttonMinusDown         = loadImage(properties.getProperty("buttonMinusDown"));
  buttonMinusX            = Integer.parseInt(properties.getProperty("buttonMinusX"));
  buttonMinusY            = Integer.parseInt(properties.getProperty("buttonMinusY"));
  buttonSliderImage       = loadImage(properties.getProperty("buttonSliderImage"));
  buttonSliderX           = Integer.parseInt(properties.getProperty("buttonSliderX"));
  buttonSliderY           = Integer.parseInt(properties.getProperty("buttonSliderY"));
  buttonQuoteImage        = loadImage(properties.getProperty("buttonQuoteImage"));
  buttonQuoteDown         = loadImage(properties.getProperty("buttonQuoteDown"));
  buttonQuoteEspImage     = loadImage(properties.getProperty("buttonQuoteEspImage"));
  buttonQuoteEspDown      = loadImage(properties.getProperty("buttonQuoteEspDown"));
  buttonQuoteGrey         = loadImage(properties.getProperty("buttonQuoteGrey"));
  buttonBiographyImage    = loadImage(properties.getProperty("buttonBiographyImage"));
  buttonBiographyDown     = loadImage(properties.getProperty("buttonBiographyDown"));
  buttonBiographyEspImage = loadImage(properties.getProperty("buttonBiographyEspImage"));
  buttonBiographyEspDown  = loadImage(properties.getProperty("buttonBiographyEspDown"));
  buttonBiographyGrey     = loadImage(properties.getProperty("buttonBiographyGrey"));
  balloonImage            = loadImage(properties.getProperty("balloonImage"));
  balloonDown             = loadImage(properties.getProperty("balloonDown"));
  sliderBarSegmentImage   = loadImage(properties.getProperty("sliderBarSegmentImage"));
  sliderBarSegmentDown    = loadImage(properties.getProperty("sliderBarSegmentDown"));
  sliderBarLeftImage      = loadImage(properties.getProperty("sliderBarLeftImage"));
  sliderBarLeftDown       = loadImage(properties.getProperty("sliderBarLeftDown"));
  sliderBarRightImage     = loadImage(properties.getProperty("sliderBarRightImage"));
  sliderBarRightDown      = loadImage(properties.getProperty("sliderBarRightDown"));
  
  /*
  imageLanguageInstructionsImage  = loadImage(properties.getProperty("imageLanguageInstructionsImage"));                  // image properties
  imageLanguageInstructionsX      = Integer.parseInt(properties.getProperty("imageLanguageInstructionsX"));
  imageLanguageInstructionsY      = Integer.parseInt(properties.getProperty("imageLanguageInstructionsY"));
  imageViewingInstructionsImage   = loadImage(properties.getProperty("imageViewingInstructionsImage"));
  imageViewingInstructionsX       = Integer.parseInt(properties.getProperty("imageViewingInstructionsX"));
  imageViewingInstructionsY       = Integer.parseInt(properties.getProperty("imageViewingInstructionsY"));
  imageZoomTitleImage             = loadImage(properties.getProperty("imageZoomTitleImage"));
  imageZoomTitleX                 = Integer.parseInt(properties.getProperty("imageZoomTitleX"));
  imageZoomTitleY                 = Integer.parseInt(properties.getProperty("imageZoomTitleY"));
  imageTitleImage                 = loadImage(properties.getProperty("imageTitleImage"));
  imageTitleX                     = Integer.parseInt(properties.getProperty("imageTitleX"));
  imageTitleY                     = Integer.parseInt(properties.getProperty("imageTitleY"));
  */
  backgroundImage                 = loadImage(properties.getProperty("backgroundImage"));
  leftFade                        = loadImage(properties.getProperty("leftFade"));
  rightFade                       = loadImage(properties.getProperty("rightFade"));
}

public void createQuote(Author author){
  ArrayList quoteLines = new ArrayList();
  int wordCount = 0;
  int lineCount = 0;
  
  String quote = "";
  if(userLanguage == "English"){
    quote = trim(author.quote_english);
  } else {
    quote = trim(author.quote_spanish);
  }
  if(quote.startsWith("\"")){
    quote = quote.substring(1);
  }
  if(quote.endsWith("\"")){
    quote = quote.substring(0, quote.length()-1);
  }
  //quote = "\"" + quote + "\"";
  quote = "“" + quote + "”";
  
  float lineHeight = author.getTextRenderer().getHeight(quote) * quoteTextScale;
  float maxLineWidth = 0;
  
  if(quote.split("\\*").length > 1){  // PREFORMATED
    String[] lines = quote.split("\\*");
    for(int i=0; i<lines.length; i++){
      trim(lines[i]);
      float lineWidth = author.getTextRenderer().getWidth(lines[i]) * quoteTextScale;  // MUST USE QUOTE TEXT RENDERER
      //println("line width: "+lineWidth);
      if(lineWidth > maxLineWidth){
        maxLineWidth = lineWidth;
      }
      float xpos = (author.getX() - (author.getMaxWidth()*0.5)) + (lineWidth*0.5);  // left aligned
      float ypos = author.getY();
      float appliedQuotationOffset = 0;
      if(i == 0){
        appliedQuotationOffset = quotationOffset;
      }
      
      QuoteLine quoteLine = new QuoteLine(numTextBlocks, author.getID(), lineCount, author, lines[i], xpos, ypos, quoteFontName, quoteFontSize, quoteTextScale, appliedQuotationOffset);
      quoteLine.setRed(quoteTextRedVal);
      quoteLine.setGreen(quoteTextGreenVal);
      quoteLine.setBlue(quoteTextBlueVal);
      quoteLine.setIntroDelay(quoteIntroDelay + (quoteFadeInDuration*lineCount));
      quoteLine.setFadeInDuration(quoteFadeInDuration);
      quoteLine.setFadeOutDuration(quoteFadeOutDuration);
      quoteLine.setStageWidth(client.getMWidth());
      quoteLine.setStageHeight(client.getMHeight());
      quoteLine.setPushMultiplier(quotePushMultiplier);
      quoteLine.clearArea(textBlocks, clearAreaMultiplier);
      
      textBlocks.put(numTextBlocks, quoteLine);
      quoteObjects.put(numTextBlocks, quoteLine);
      quoteLines.add(quoteLine);
      numTextBlocks++;
      lineCount++;
    }
  } else {                            // AUTOFORMATED
    String[] words = quote.split(" ");
    while(wordCount < words.length){
      int charCount = 0;
      String textLine = "";
      while(charCount < charsPerLine){
        if(wordCount < words.length){
          charCount += words[wordCount].length() + 1;
          textLine += words[wordCount] + " ";
          wordCount++;
        } else {
          break;
        }
      }
      trim(textLine);  // remove excess white space from back of line
      float lineWidth = author.getTextRenderer().getWidth(textLine) * quoteTextScale;  // MUST USE QUOTE TEXT RENDERER
      //println("line width: "+lineWidth);
      if(lineWidth > maxLineWidth){
        maxLineWidth = lineWidth;
      }
      //float xpos = random(author.getX() - (lineWidth*0.5), author.getX());  // random positioning
      //float xpos = (author.getX() - (author.getWidth()*0.5)) + (lineWidth*0.5);  // left aligned
      float xpos = (author.getX() - (author.getMaxWidth()*0.5)) + (lineWidth*0.5);  // left aligned
      float ypos = author.getY();
      
      float appliedQuotationOffset = 0;
      if(lineCount == 0){
        appliedQuotationOffset = quotationOffset;
      }
      
      QuoteLine quoteLine = new QuoteLine(numTextBlocks, author.getID(), lineCount, author, textLine, xpos, ypos, quoteFontName, quoteFontSize, quoteTextScale, appliedQuotationOffset);
      quoteLine.setRed(quoteTextRedVal);
      quoteLine.setGreen(quoteTextGreenVal);
      quoteLine.setBlue(quoteTextBlueVal);
      quoteLine.setIntroDelay(quoteIntroDelay + (quoteFadeInDuration*lineCount));
      quoteLine.setFadeInDuration(quoteFadeInDuration);
      quoteLine.setFadeOutDuration(quoteFadeOutDuration);
      quoteLine.setStageWidth(client.getMWidth());
      quoteLine.setStageHeight(client.getMHeight());
      quoteLine.setPushMultiplier(quotePushMultiplier);
      quoteLine.clearArea(textBlocks, clearAreaMultiplier);
      
      textBlocks.put(numTextBlocks, quoteLine);
      quoteObjects.put(numTextBlocks, quoteLine);
      quoteLines.add(quoteLine);
      numTextBlocks++;
      lineCount++;
    }
  }
  
  author.setFadeInDuration(quoteFadeInDuration);  
  author.setFadeOutDuration(quoteFadeOutDuration);
  author.setAuthorFadeOutDelay(authorFadeOutDelay);
  author.setHoldDuration((quoteHoldDuration*lineCount) + quoteIntroDelay + authorFadeOutDelay);

  ArrayList textBlocksToRemove = new ArrayList();  // list of ID numbers for textblocks to remove 
 
  // MUST FIND CENTER LOCATION OF THE ENTIRE QUOTE BLOCK
  float leftmost, rightmost, topmost, bottommost;  // top left and bottom right corners
  float centerX, centerY;                          // center of text block
  if(author.getY() < client.getMHeight()*0.5){
    if(author.getHeight() > author.getMaxHeight()){
      leftmost = author.getX() - (author.getWidth()*0.5);    // left side of author name and quote block
      topmost = author.getY() + (author.getHeight()*0.5);    // bottom edge of author text, top edge of quote block
    } else {
      leftmost = author.getX() - (author.getMaxWidth()*0.5); // left side of author name and quote block
      topmost = author.getY() + (author.getMaxHeight()*0.5); // bottom edge of author text, top edge of quote block
    }
    rightmost = leftmost + maxLineWidth;                   // farthest right of left aligned location 
    bottommost = topmost + (lineHeight * lineCount);       // farthest down from bottom of author name
  } else {
    if(author.getHeight() > author.getMaxHeight()){
      leftmost = author.getX() - (author.getWidth()*0.5);    // left side of author name and quote block
      bottommost = author.getY() - (author.getHeight()*0.5); // top edge of author text, bottom edge of quote block
    } else {
      leftmost = author.getX() - (author.getMaxWidth()*0.5);    // left side of author name and quote block
      bottommost = author.getY() + (author.getMaxHeight()*0.5); // top edge of author text, bottom edge of quote block
    }
    rightmost = leftmost + maxLineWidth;                  // farthest right of left aligned location
    topmost = bottommost - (lineHeight * lineCount);      // top edge of top quote line
  }
  
  centerX = leftmost + ((rightmost - leftmost)/2);
  centerY = topmost + ((bottommost - topmost)/2);
  
  for(int i=0; i<quoteLines.size(); i++){
    QuoteLine quoteLine = ((QuoteLine)quoteLines.get(i));
    //Author quoteAuthor = quoteLine.getAuthor();
    //quoteLine.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*i)));  // this fades all lines out at once
    quoteLine.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*(lineCount-i))));    // this fades lines out as they came in
    //println("centerX: "+centerX +" centerY: "+ centerY);
    //println("left side: "+leftmost+" right side: "+rightmost);
    quoteLine.setParagraphCenter(centerX, centerY);
    quoteLine.setParagraphDimensions(rightmost - leftmost, bottommost - topmost);
    
    float ypos;
    if(author.getY() < client.getMHeight()*0.5){
      //ypos = (lineHeight * i) + ((lineHeight*0.5) + author.getY()+(author.getHeight()*0.5) + textMarginVertical+1);
      if(author.getHeight() > author.getMaxHeight()){
        ypos = (lineHeight * i) + ((lineHeight*0.5) + author.getY()+(author.getHeight()*0.5) + textMarginVertical+1);
        ypos += quoteBlockTopMargin;
      } else {
        ypos = (lineHeight * i) + ((lineHeight*0.5) + author.getY()+(author.getMaxHeight()*0.5) + textMarginVertical+1);
        ypos += quoteBlockTopMargin;
      }
    } else {
      //ypos = (author.getY()-(author.getHeight()*0.5)) - ((lineCount*lineHeight) - (lineHeight*0.5) - (lineHeight * i) + (textMarginVertical+1));
      if(author.getHeight() > author.getMaxHeight()){
        ypos = (author.getY() - (author.getHeight()*0.5)) - ((lineCount*lineHeight) - (lineHeight*0.5) - (lineHeight * i) + (textMarginVertical+1));
        ypos -= quoteBlockTopMargin;
      } else {
        ypos = (author.getY() - (author.getMaxHeight()*0.5)) - ((lineCount*lineHeight) - (lineHeight*0.5) - (lineHeight * i) + (textMarginVertical+1));
        ypos -= quoteBlockTopMargin;
      }
    }
    //println(ypos);
    quoteLine.setY(ypos);
    
    // CHECK SURROUNDING AREA FOR ANY OTHER QUOTES WITHIN THE THRESHOLD THAT SHOULD BE REMOVED
    Iterator iter = quoteObjects.values().iterator();
    while(iter.hasNext()){
      QuoteLine otherQuoteLine = (QuoteLine)iter.next();
      if(otherQuoteLine.getQuoteID() != quoteLine.getQuoteID()){
        float xdist = 0;
        float ydist = 0;
        if(otherQuoteLine.getX() > quoteLine.getX()){
          xdist = (otherQuoteLine.getX() - (otherQuoteLine.getWidth()/2)) - (quoteLine.getX() + (quoteLine.getWidth()/2));
        } else {
          xdist = (quoteLine.getX() - (quoteLine.getWidth()/2)) - (otherQuoteLine.getX() + (otherQuoteLine.getWidth()/2));
        }
        if(xdist < 0){
          xdist = 0;
        }
        
        if(otherQuoteLine.getY() > quoteLine.getY()){
          ydist = (otherQuoteLine.getY() - (otherQuoteLine.getHeight()/2)) - (quoteLine.getY() + (quoteLine.getHeight()/2));
        } else {
          ydist = (quoteLine.getY() - (quoteLine.getHeight()/2)) - (otherQuoteLine.getY() + (otherQuoteLine.getHeight()/2));
        }
        if(ydist < 0){
          ydist = 0;
        }
        
        float hypo = sqrt(sq(xdist) + sq(ydist));
        println("hypo: "+hypo +" xdist: "+ xdist +" ydist: "+ ydist + " threshold: "+ quoteDistanceThreshold);
        if(hypo < quoteDistanceThreshold){
          textBlocksToRemove.add(otherQuoteLine.getQuoteID());
        }
      }
    }
    
    // CHECK SURROUNDING AREA FOR ANY OTHER QUOTES/BIOGRAPHIES WITHIN THE THRESHOLD THAT SHOULD BE REMOVED
    iter = bioObjects.values().iterator();
    while(iter.hasNext()){
      QuoteLine otherQuoteLine = (QuoteLine)iter.next();
      if(otherQuoteLine.getQuoteID() != quoteLine.getQuoteID()){
        float xdist = 0;
        float ydist = 0;
        if(otherQuoteLine.getX() > quoteLine.getX()){
          xdist = (otherQuoteLine.getX() - (otherQuoteLine.getWidth()/2)) - (quoteLine.getX() + (quoteLine.getWidth()/2));
        } else {
          xdist = (quoteLine.getX() - (quoteLine.getWidth()/2)) - (otherQuoteLine.getX() + (otherQuoteLine.getWidth()/2));
        }
        if(xdist < 0){
          xdist = 0;
        }
        
        if(otherQuoteLine.getY() > quoteLine.getY()){
          ydist = (otherQuoteLine.getY() - (otherQuoteLine.getHeight()/2)) - (quoteLine.getY() + (quoteLine.getHeight()/2));
        } else {
          ydist = (quoteLine.getY() - (quoteLine.getHeight()/2)) - (otherQuoteLine.getY() + (otherQuoteLine.getHeight()/2));
        }
        if(ydist < 0){
          ydist = 0;
        }
        
        float hypo = sqrt(sq(xdist) + sq(ydist));
        println("hypo: "+hypo +" xdist: "+ xdist +" ydist: "+ ydist + " threshold: "+ quoteDistanceThreshold);
        if(hypo < quoteDistanceThreshold){
          textBlocksToRemove.add(otherQuoteLine.getQuoteID());
        }
      }
    }
    
  }
  
  for(int i=0; i<textBlocksToRemove.size(); i++){
    int authorID = (Integer)textBlocksToRemove.get(i);
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock textBlock = (TextBlock)iter.next();
      if(textBlock instanceof QuoteLine){
        if(((QuoteLine)textBlock).getQuoteID() == authorID){
          textBlock.fadeOutAndRemove();
        }
      } else if(textBlock instanceof Author){
        if(textBlock.getID() == authorID){
          textBlock.fadeOutAndRemove(authorFadeOutDelay);
        }
      }
    }
  }
  
  
  
  // check each one for the oldest block of text
  int oldest = Integer.MAX_VALUE;  // check each ID for lowest value
  int oldestID = 0;            // author/quote/bio shared ID
  String oldestType = "";
  ConcurrentHashMap combinedBlocks = new ConcurrentHashMap();
  Iterator iter = quoteObjects.values().iterator();  // check all quotes first
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    combinedBlocks.put(((QuoteLine)textBlock).getQuoteID(), true);  // store quote ID as key
    if(textBlock.getID() < oldest){
      oldest = textBlock.getID();
      oldestID = ((QuoteLine)textBlock).getQuoteID();
      oldestType = "Quote";
    }
  }
  iter = bioObjects.values().iterator();  // check bios
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    combinedBlocks.put(((QuoteLine)textBlock).getQuoteID(), true);  // store quote ID as key
    if(textBlock.getID() < oldest){
      oldest = textBlock.getID();
      oldestID = ((QuoteLine)textBlock).getQuoteID();
      oldestType = "Bio";
    }
  }
  
  if(combinedBlocks.size() > maxQuotes){
    if(oldestType.equals("Bio")){
      iter = bioObjects.values().iterator();  // check bios
      while(iter.hasNext()){
        QuoteLine textBlock = (QuoteLine)iter.next();
        if(textBlock.getQuoteID() == oldestID){
          textBlock.fadeOutAndRemove();
        }
      }
    } else if(oldestType.equals("Quote")){
      iter = quoteObjects.values().iterator();  // check bios
      while(iter.hasNext()){
        QuoteLine textBlock = (QuoteLine)iter.next();
        if(textBlock.getQuoteID() == oldestID){
          textBlock.fadeOutAndRemove();
        }
      }
    }
    ((Author)authorObjects.get(oldestID)).fadeOutAndRemove(authorFadeOutDelay);  // deactivate the author
  }
  
}

void createBio(Author author){
  ArrayList bioLines = new ArrayList();
  int wordCount = 0;
  int lineCount = 0;
  String bio = "";
  if(userLanguage == "English"){
    bio = trim(author.bio_english);
  } else {
    bio = trim(author.bio_spanish);
  }
  float lineHeight = author.getTextRenderer().getHeight(bio) * quoteTextScale;  // TODO: this will not be precise if not using the quote text renderer
  String[] words = bio.split(" ");
  
  // create the genre text object that will go under the author name
  String genreString = join(author.genres, ", ").toUpperCase();
  float genreX = author.getX() + (author.getWidth()/2);
  float genreY = author.getY();
  QuoteLine genreObj = new QuoteLine(numTextBlocks, author.getID(), 0, author, genreString, genreX, genreY, genreFontName, genreFontSize, genreTextScale, 0);
  float genreHeight = genreObj.getTextRenderer().getHeight(genreString) * genreTextScale;
  /*
  // THIS SNAPS GENRE TO TOP OR BOTTOM OF AUTHOR NAME
  if(author.getY() < client.getMHeight()*0.5){
    genreY = genreY + (author.getHeight()/2) + genreHeight/2;
  } else {
    genreY = genreY - (author.getHeight()/2) - genreHeight/2;
  }
  genreObj.setY(genreY);
  */
  genreObj.setRed(genreTextRedVal);
  genreObj.setGreen(genreTextGreenVal);
  genreObj.setBlue(genreTextBlueVal);
  genreObj.setIntroDelay(0);
  genreObj.setFadeInDuration(quoteFadeInDuration);
  genreObj.setFadeOutDuration(quoteFadeOutDuration);
  genreObj.setStageWidth(client.getMWidth());
  genreObj.setStageHeight(client.getMHeight());
  genreObj.clearArea(textBlocks, clearAreaMultiplier);
  
  textBlocks.put(numTextBlocks, genreObj);
  bioObjects.put(numTextBlocks, genreObj);
  numTextBlocks++;
  
  // create the date text object that will go on the right of the author name
  String dateString = "b.";
  if(author.born > 0){
    dateString += author.born;
  }
  if(author.died > 0){
    dateString += "-" + author.died;
  }
  float dateX = author.getX();
  if(author.getScale() < activatedScaleMax){
    dateX += (author.getMaxWidth()/2);
  } else {
    dateX += (author.getWidth()/2);
  }
  float dateY = author.getY();
  QuoteLine dateObj = new QuoteLine(numTextBlocks, author.getID(), 0, author, dateString, dateX, dateY, dateFontName, dateFontSize, dateTextScale, 0);
  float dateHeight = dateObj.getTextRenderer().getHeight(dateString) * dateTextScale; 
  if(author.getY() < client.getMHeight()*0.5){
    dateY = dateY + (author.getHeight()/2) - dateHeight/2;
  } else {
    dateY = dateY - (author.getHeight()/2) + dateHeight/2;
  }
  dateX += (dateObj.getWidth()/2) + dateTextLeftMargin;
  dateObj.setX(dateX);
  dateObj.setY(dateY);
  dateObj.setRed(dateTextRedVal);
  dateObj.setGreen(dateTextGreenVal);
  dateObj.setBlue(dateTextBlueVal);
  dateObj.setIntroDelay(0);
  dateObj.setFadeInDuration(quoteFadeInDuration);
  dateObj.setFadeOutDuration(quoteFadeOutDuration);
  dateObj.setStageWidth(client.getMWidth());
  dateObj.setStageHeight(client.getMHeight());
  dateObj.clearArea(textBlocks, clearAreaMultiplier);
  dateObj.snapToRight(dateTextLeftMargin);
  
  if(author.born > 0){
    textBlocks.put(numTextBlocks, dateObj);
    bioObjects.put(numTextBlocks, dateObj);
    numTextBlocks++;
  }
  
    
  // process the bio paragraph into separate lines
  while(wordCount < words.length){
    int charCount = 0;
    String textLine = "";
    while(charCount < charsPerLine){
      if(wordCount < words.length){
        charCount += words[wordCount].length() + 1;
        textLine += words[wordCount] + " ";
        wordCount++;
      } else {
        break;
      }
    }
    trim(textLine);  // remove excess white space from back of line
    float lineWidth = author.getTextRenderer().getWidth(textLine) * quoteTextScale;
    float xpos = (author.getX() - (author.getWidth()*0.5)) + (lineWidth*0.5);  // left aligned
    float ypos = author.getY();
    
    QuoteLine bioLine = new QuoteLine(numTextBlocks, author.getID(), lineCount, author, textLine, xpos, ypos, bioFontName, bioFontSize, bioTextScale, 0);
    bioLine.setRed(bioTextRedVal);
    bioLine.setGreen(bioTextGreenVal);
    bioLine.setBlue(bioTextBlueVal);
    bioLine.setIntroDelay(quoteIntroDelay + (quoteFadeInDuration*lineCount));
    bioLine.setFadeInDuration(quoteFadeInDuration);
    bioLine.setFadeOutDuration(quoteFadeOutDuration);
    bioLine.setStageWidth(client.getMWidth());
    bioLine.setStageHeight(client.getMHeight());
    bioLine.clearArea(textBlocks, clearAreaMultiplier);
    
    textBlocks.put(numTextBlocks, bioLine);
    bioObjects.put(numTextBlocks, bioLine);
    bioLines.add(bioLine);
    numTextBlocks++;
    lineCount++;
  }
  author.setFadeInDuration(quoteFadeInDuration);
  author.setFadeOutDuration(quoteFadeOutDuration);
  author.setHoldDuration((quoteHoldDuration*lineCount) + quoteIntroDelay + authorFadeOutDelay);
  author.setAuthorFadeOutDelay(authorFadeOutDelay);
  genreObj.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*lineCount)));
  dateObj.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*lineCount)));
  
  if(author.getY() < client.getMHeight()*0.5){
    genreY = genreY + (author.getHeight()/2) + genreHeight/2;
    genreY += genreTextTopMargin;
  } else {
    genreY = genreY - (author.getHeight()/2) - genreHeight/2 - (lineCount * lineHeight);
    genreY -= genreTextTopMargin;
  }
  genreObj.setY(genreY);
  
  for(int i=0; i<bioLines.size(); i++){
    QuoteLine bioLine = ((QuoteLine)bioLines.get(i));
    //bioLine.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*i)));
    bioLine.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*(lineCount-i))));    // this fades lines out as they came in
    float ypos;
    if(author.getY() < client.getMHeight()*0.5){
      ypos = (lineHeight * i) + ((lineHeight*0.5) + author.getY()+(author.getHeight()*0.5) + textMarginVertical+1);
      ypos += genreHeight;
      ypos += bioBlockTopMargin;
    } else {
      ypos = (author.getY()-(author.getHeight()*0.5)) - ((lineCount*lineHeight) - (lineHeight*0.5) - (lineHeight * i) + (textMarginVertical+1));
      //ypos -= genreHeight;
      ypos -= bioBlockTopMargin;      
    }
    bioLine.setY(ypos);
  }
  
  // TODO: CHECK SURROUNDING AREA FOR ANY OTHER QUOTES/BIOGRAPHIES WITHIN THE THRESHOLD THAT SHOULD BE REMOVED
  
}






// FUNCTIONS FOR CHANGING VISUAL MODES

void authorCloudMode(){
  Iterator iter = authorObjects.values().iterator();
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    author.moveTo(random(0, client.getMWidth()), random(0, client.getMHeight()), 60);
  }
}

void sortByGenre(String genre){
  Iterator iter = authorObjects.values().iterator();
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    Boolean genrematch = false;
    for(int i=0; i<author.genres.length; i++){
      if(author.genres[i].equals(genre)){
        // move author name on to the screen
        float xpos = random(0, client.getMWidth());
        float ypos = random(0, 768);
        //println(author.getText() +" "+ genre +" "+ xpos);
        author.moveTo(xpos, ypos, 60);  // TODO: switch to props file duration
        genrematch = true;
        break;
      }
    }
    if(!genrematch){
      // send author name off the screen
      if(random(0,1) > 0.5){
        author.moveTo(random(0, client.getMWidth()), random(client.getMHeight(), client.getMHeight()+1000), 60);
      } else {
        author.moveTo(random(0, client.getMWidth()), random(-1000, 0), 60);
      }
    }
  }
}

void sortByDate(){
  int highest = 0;
  int lowest = 10000;
  Iterator iter = authorObjects.values().iterator();
  // loop through all authors and find the range of dates
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    // SHOULD DEFINE WHAT PROPERTY TO DO THE ANALYSIS ON
    if(author.workbegan > highest){
      highest = author.workbegan;
    } else if(author.workbegan < lowest && author.workbegan > 0){
      lowest = author.workbegan;
    }
  }
  float range = highest - lowest;
  // loop through all authors AGAIN and place them on the stage according to their date
  iter = authorObjects.values().iterator();
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    float xpos = ((author.workbegan - lowest)/range) * client.getMWidth();
    author.moveTo(xpos, random(0,768), 60);
    println(author.getText() +" "+ author.workbegan +" "+ xpos);
  }
}

void sortByPopularity(){
  Iterator iter = authorObjects.values().iterator();
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    float xpos = (author.popularity * 0.01) * client.getMWidth();
    author.moveTo(xpos, random(0,768), 60);
    println(author.getText() +" "+ author.popularity +" "+ xpos);
  }
}

void resetCloud(){
  // (1) change all author positions and damp to their new target
  // (2) change all author text scales and tween to their new size
  Iterator iter = authorObjects.values().iterator();
  int authorCounter = 0;
  while(iter.hasNext()){
    float textscale = random(textScaleMin, textScaleMin + (((100-authorCounter) * 0.01) * (textScaleMax-textScaleMin)));
    Author author = (Author)iter.next();
    //author.moveTo(random(0-overflow,client.getMWidth()+overflow), random(0,client.getMHeight()), resetDuration);
    author.moveTo(random(0-overflow,client.getMWidth()+overflow), random(312,768), resetDuration);
    author.scaleTo(textscale, resetDuration);
    authorCounter++;
  }
}





// RENDER THREAD

void render(TCPClient c){
  background(backgroundGray);
  if(displayControls){
    image(backgroundImage, 0, 0);
  }
  
  // calculate zooming of interfaceScale
  if(zooming){
    float progress = sin((zoomCounter / (float)zoomDuration) * (PI/2));
    interfaceScale = zoomStart + ((zoomTarget-zoomStart) * progress);
    if(interfaceScale > minZoom){
      interfaceScale = minZoom;
    } else if(interfaceScale < maxZoom){
      interfaceScale = maxZoom;
    }
    scaledWidth = client.getMWidth() * (1/interfaceScale);
    scaledHeight = client.getMHeight() * (1/interfaceScale);
    horizontalMouseOffset = (scaledWidth/2) - (width/2 + horizontalOffset);      // centered
    verticalMouseOffset = (scaledHeight/2) - (height/2 + verticalOffset);         // centered
    slider.setAreaVisible(width/(float)scaledWidth);
    slider.setOffset(horizontalMouseOffset/(float)scaledWidth);
    // check slider to make sure we aren't zooming out beyond the allowed viewable area
    float offset = 0 - (slider.getBarPosition() - 0.5);  // cloud is centered
    horizontalOffset = int(offset * scaledWidth);
    if(balloon != null){
      balloon.setInterfaceScale(interfaceScale);
      balloon.setHorizontalOffset(horizontalMouseOffset);
      balloon.setVerticalOffset(verticalMouseOffset);
    }
    zoomCounter++;
    if(zoomCounter == zoomDuration){
      zooming = false;
      zoomCounter = 0;
      if(displayControls){
        btnPlus.silentOff();
        btnMinus.silentOff();
      }
    }
  }
  
  pushMatrix();
  
  // THIS IS WHERE ALL THE SCALING AND TRANSLATING OF THE INTERFACE OCCURS, AND SHOULD NOT AFFECT PROJECTED VERSION
  if(enableCamera){
    //scale(1/interfaceScale);      // THIS IS THE STATIC METHOD (ie: no zooming)
    //translate(0,verticalOffset,0);
    translate(width/2 + horizontalOffset, height/2 + verticalOffset, 0);    // translates the center
    pushMatrix();
    scale(1/interfaceScale);                                                // scales the content to fit the screen
    translate(0-(client.getMWidth()/2), 0-(client.getMHeight()/2), 0);      // translates content to center over 0,0
    // THESE CALCULATIONS NEED TO STAY FOR WHEN THE TWEENED ZOOM EVENTS OCCUR
    scaledWidth = client.getMWidth() * (1/interfaceScale);
    scaledHeight = client.getMHeight() * (1/interfaceScale);
    horizontalMouseOffset = (scaledWidth/2) - (width/2 + horizontalOffset);      // centered
    verticalMouseOffset = (scaledHeight/2) - (height/2 + verticalOffset);         // centered
    //println(horizontalMouseOffset + " "+ verticalMouseOffset);
    //ellipse((mouseX + horizontalMouseOffset) * interfaceScale, (mouseY + verticalMouseOffset) * interfaceScale, 10, 10);
    if(balloon != null){
      balloon.setInterfaceScale(interfaceScale);
      balloon.setHorizontalOffset(horizontalMouseOffset);
      balloon.setVerticalOffset(verticalMouseOffset);
    }
  }
 
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.render(pgl, c.getXoffset(), c.getYoffset(), yflip);
    if(displayBoundingBoxes){
      textBlock.drawBoundingBox(c.getXoffset(), c.getYoffset());
    }
  }
  
  iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.move(c.getMWidth(), overflow);
  }
  
  if(mode == "authorcloud"){
    iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock textBlock = (TextBlock)iter.next();
      textBlock.applyGravity(xGrav, yGrav, xgravity, ygravity);
    }
  }
  
  iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    if(textBlock instanceof QuoteLine){
      textBlock.checkCollisions(textBlocks, quoteMarginHorizontal, quoteMarginVertical);
    } else {
      textBlock.checkCollisions(textBlocks, textMarginHorizontal, textMarginVertical);
    }
  }
  
  // check all text blocks to see if they should be removed or not
  iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    if(textBlock.remove){
      //println("removing "+ textBlock.getText());
      textBlocks.remove(textBlock.id);
      if(textBlock instanceof QuoteLine){
        quoteObjects.remove(textBlock.id);
        bioObjects.remove(textBlock.id);
        // apply attractor vector to surrounding authors to make them swarm back in
        float attractorVal = 2;
        int distance = 100;
        Iterator innerIter = textBlocks.values().iterator();
        while(innerIter.hasNext()){
          TextBlock otherBlock = (TextBlock)innerIter.next();
          //println(otherBlock.getText());
          if(textBlock.getID() != otherBlock.getID()){
            if((otherBlock.getY()+(otherBlock.getHeight()*0.5) > textBlock.getY() - (textBlock.getHeight()*0.5)) && (otherBlock.getY()-(otherBlock.getHeight()*0.5) < textBlock.getY()+(textBlock.getHeight()*0.5))){
              if(otherBlock.getX() > textBlock.getX()){
                //println((otherBlock.getX()-(otherBlock.getWidth()*0.5)) - (textBlock.getX()+(textBlock.getWidth()*0.5)));
                if((otherBlock.getX()-(otherBlock.getWidth()*0.5)) - (textBlock.getX()+(textBlock.getWidth()*0.5)) < distance){
                  //println("pulling on "+otherBlock.getText());
                  otherBlock.push(0 - attractorVal, 0);
                }
              } else {
                if((textBlock.getX()-(textBlock.getWidth()*0.5)) - (otherBlock.getX()+(otherBlock.getWidth()*0.5)) < distance){
                  //println("pulling on "+otherBlock.getText());
                  otherBlock.push(attractorVal, 0);
                }
              }
            }
          }
        }
      }
    }
  }
  
  // check all quote objects to see if they are fully off screen, and if so, remove them immediately
  iter = quoteObjects.values().iterator();
  while(iter.hasNext()){                                                // for every quote line...
    QuoteLine quoteLine = (QuoteLine)iter.next();
    if(quoteLine.getX()-(quoteLine.getWidth()/2) > client.getMWidth()){  // if quote line off screen...
      boolean removeQuote = true;
      Iterator innerIter = quoteObjects.values().iterator();
      while(innerIter.hasNext()){                                        // check other quote lines belonging to this quote
        QuoteLine otherQuoteLine = (QuoteLine)innerIter.next();
        if(otherQuoteLine.getQuoteID() == quoteLine.getQuoteID()){      // if belonging to this quote..
          if(otherQuoteLine.getX()-(otherQuoteLine.getWidth()/2) < client.getMWidth()){  // if not beyond the right edge of the screen...
            removeQuote = false;                                        // remove nothing and break inner loop
            break;
          }
        }
      }
      if(removeQuote){
        quoteLine.removeNow();
        ((Author)authorObjects.get(quoteLine.getQuoteID())).deactivate();
      }
    } else if(quoteLine.getX() + (quoteLine.getWidth()/2) < 0){
      boolean removeQuote = true;
      Iterator innerIter = quoteObjects.values().iterator();
      while(innerIter.hasNext()){                                        // check other quote lines belonging to this quote
        QuoteLine otherQuoteLine = (QuoteLine)innerIter.next();
        if(otherQuoteLine.getQuoteID() == quoteLine.getQuoteID()){      // if belonging to this quote..
          if(otherQuoteLine.getX() + (otherQuoteLine.getWidth()/2) > 0){
            removeQuote = false;                                        // remove nothing and break inner loop
            break;
          }
        }
      }
      if(removeQuote){
        quoteLine.removeNow();
        ((Author)authorObjects.get(quoteLine.getQuoteID())).deactivate();
      }
    }
  }
  
  // check all BIOGRAPHY objects to see if they are fully off screen, and if so, remove them immediately
  iter = bioObjects.values().iterator();
  while(iter.hasNext()){
    QuoteLine bioLine = (QuoteLine)iter.next();
    if(bioLine.getX()-(bioLine.getWidth()/2) > client.getMWidth()){ // if bio line off the screen...
      boolean removeBio = true;
      Iterator innerIter = bioObjects.values().iterator();
      while(innerIter.hasNext()){
        QuoteLine otherBioLine = (QuoteLine)innerIter.next();
        if(otherBioLine.getQuoteID() == bioLine.getQuoteID()){      // if belonging to this quote..
          if(otherBioLine.getX()-(otherBioLine.getWidth()/2) < client.getMWidth()){  // if not beyond the right edge of the screen...
            removeBio = false;                                        // remove nothing and break inner loop
            break;
          }
        }
      }
      if(removeBio){
        bioLine.removeNow();
        ((Author)authorObjects.get(bioLine.getQuoteID())).deactivate();
      }
    } else if(bioLine.getX() + (bioLine.getWidth()/2) < 0){
      boolean removeBio = true;
      Iterator innerIter = bioObjects.values().iterator();
      while(innerIter.hasNext()){                                        // check other quote lines belonging to this quote
        QuoteLine otherBioLine = (QuoteLine)innerIter.next();
        if(otherBioLine.getQuoteID() == bioLine.getQuoteID()){      // if belonging to this quote..
          if(otherBioLine.getX() + (otherBioLine.getWidth()/2) > 0){
            removeBio = false;                                        // remove nothing and break inner loop
            break;
          }
        }
      }
      if(removeBio){
        bioLine.removeNow();
        ((Author)authorObjects.get(bioLine.getQuoteID())).deactivate();
      }
    }
  }
  
  // GRADIENTS ON EDGES TO FADE ALL TEXT HORIZONTALLY
  image(leftFade, 0, -1000, horizontalFallOff, client.getMHeight()+2000);
  image(rightFade, client.getMWidth() - horizontalFallOff, -1000, horizontalFallOff, client.getMHeight()+2000);
  // additional black area beyond screen just in case of scaling rounding issues
  fill(0);
  rect(-20,-1000,20,client.getMHeight()+2000);
  rect(client.getMWidth(),-1000,20,client.getMHeight()+2000);
  
  if(displayFrames){
    stroke(255);
    noFill();
    rect(0,156,1024,768);       // three projection frames
    rect(1024,156,1024,768); 
    rect(2048,156,1024,768);
    stroke(255,0,0);
    rect(0,0,3072,1080);        // entire MPE stage
    line(1536, 500, 1536, 580); // cross hairs in center of stage
    line(1496, 540, 1576, 540);
    noStroke();
  }
  
  if(enableCamera){
    popMatrix();
  }
  
  popMatrix();
  
  if(displayControls){
    //image(imageLanguageInstructionsImage, imageLanguageInstructionsX, imageLanguageInstructionsY);
    //image(imageViewingInstructionsImage, imageViewingInstructionsX, imageViewingInstructionsY);
    //image(imageZoomTitleImage, imageZoomTitleX, imageZoomTitleY);
    //image(imageTitleImage, imageTitleX, imageTitleY);
    //image(buttonSliderImage, buttonSliderX, buttonSliderY);  // TEMPORARY until widget object is created
    widgetManager.draw();
  }
  
  textFont(statFont);
  fill(200);
  if(displayStats){
    pushMatrix();
    translate(client.getXoffset(), client.getYoffset());
    text(int(frameRate) +" fps", 10, 20);
    
    if(System.currentTimeMillis() - lastTime > 0){
      float mpeDelay = 1000.0 / (System.currentTimeMillis() - lastTime);  // 1 second divided by duration since last frame in ms
      mpeFpsHistory.add(mpeDelay);
    }
    if(mpeFpsHistory.size() == 30){
      float mpeTotal = 0;
      for(int i=0; i<mpeFpsHistory.size(); i++){
        mpeTotal += (Float)mpeFpsHistory.get(i);
      }
      mpeFps = int(mpeTotal/mpeFpsHistory.size());
      mpeFpsHistory.remove(0);
    }
    
    text(mpeFps + " mpe fps", 10, 35);
    text(textBlocks.size() + " total text objects", 10, 50);
    text(authorObjects.size() +" author names", 10, 65);
    text(quoteObjects.size() +" quote lines", 10, 80);
    text(bioObjects.size() +" bio lines", 10, 95);
    lastTime = System.currentTimeMillis();  // record time for FPS comparison
    popMatrix();
  }
  
  inactivityCounter++;  // keep counting forever, gets reset on mousePressedEvent
  if(inactivityCounter > screensaverInactivity){
    screensaverActivated = true;
  } else {
    screensaverActivated = false;
    resetCounter = 0;
    zoomDelayCounter = 0;
  }
  
  if(screensaverActivated){
    iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock textBlock = (TextBlock)iter.next();
      textBlock.xv += screensaverSpeed;
    }
    
    /*
    // THIS IS WHERE FUCKED UP INACTIVITY ZOOMING OCCURS
    if(interfaceScale != defaultInterfaceScale){
      if(!zooming){
        zoomDelayCounter++;
      }
      if(zoomDelayCounter > inactivityZoomDelay){
        zoomDuration = inactivityZoomDuration;
        zoomTarget = defaultInterfaceScale;
        zoomStart = interfaceScale;
        zoomDelayCounter = 0;
        zooming = true;
      }
    }
    */
    
    resetCounter++;
    if(resetCounter > resetDelay){
      // trigger "freak out" and re-arrange all author names as well as randomize and tween to a new textscale
      resetCloud();
      resetCounter = 0;
    }
  }
  
}

void draw(){
  //if(standAlone){
  //  render(client);
  //}
}

void frameEvent(TCPClient c){
  // TODO: this will take over for draw() when MPE is implemented.
  if(c.messageAvailable()){
    String[] msg = c.getDataMessage();
    //println(msg[0]);
    String[] command = msg[0].split(",");
    if(command[0].equals("drag")){    // used for roll overs and moving the text cloud side to side
      Iterator iter = textBlocks.values().iterator();
      while(iter.hasNext()){
        TextBlock textBlock = (TextBlock)iter.next();
        textBlock.xv += Integer.parseInt(command[1])*dragDamp;
        float distance = abs(textBlock.y - Integer.parseInt(command[3]));
        if(distance < dragRadius){
          textBlock.xv += Integer.parseInt(command[1]) * (dragDamp * (1 - (distance/dragRadius)));
        }
        if(textBlock.isOver(Integer.parseInt(command[2]), Integer.parseInt(command[3]))){
          textBlock.rollOver();
        } else {
          textBlock.rollOut();
        }
      }
      lastDragged = System.currentTimeMillis();
    } else if(command[0].equals("press")){    // used to register intentional contact with author names
      mousePressedEvent(Integer.parseInt(command[1]), Integer.parseInt(command[2]));
    } else if(command[0].equals("release")){  // used to check for intentional click of author names
      mouseReleasedEvent(Integer.parseInt(command[1]), Integer.parseInt(command[2]));
    } else if(command[0].equals("buttonEvent")){
      if(command[1].equals("english")){
        userLanguage = "English";
        if(displayControls){
          btnEspanol.silentOff();
          if(balloon != null){
            balloon.setUserLanguage(userLanguage);
          }
        }
      } else if(command[1].equals("espanol")){
        userLanguage = "Espanol";
        if(displayControls){
          btnEnglish.silentOff();
          if(balloon != null){
            balloon.setUserLanguage(userLanguage);
          }
        }
      } else if(command[1].equals("zoomin")){
        if(enableCamera){
          if(interfaceScale > maxZoom){
            // TODO: tween the interfaceScale down by 0.1
            zoomDuration = buttonZoomDuration;
            zoomTarget = interfaceScale - 0.1;
            zoomStart = interfaceScale;
            zooming = true;
            zoomCounter = 0;
          } else {
            btnPlus.silentOff();
          }
        }
      } else if(command[1].equals("zoomout")){
        if(enableCamera){
          if(interfaceScale < minZoom){
            // TODO: tween the interfaceScale up by 0.1
            zoomDuration = buttonZoomDuration;
            zoomTarget = interfaceScale + 0.1;
            zoomStart = interfaceScale;
            zooming = true;
            zoomCounter = 0;
          } else {
            btnMinus.silentOff();
          }
        }
      } else if(command[1].equals("slide")){
        if(enableCamera){
          float offset = 0 - (Float.parseFloat(command[2]) - 0.5);  // cloud is centered
          horizontalOffset = int(offset * scaledWidth);
          //println(offset +" horizontalOffset: "+ horizontalOffset + " horizontalMouseOffset: "+ horizontalMouseOffset);
        }
      } else if(command[1].equals("cloud")){
        if(command[2].equals("normal")){
          if(displayControls){
            btnDate.silentOff();
            dropdownGenre.silentOff();
            btnPopularity.silentOff();
          }
          mode = "authorcloud";
          // TODO: cause all author names to condense towards the middle of the screen
          authorCloudMode();
        } else if(command[2].equals("date")){
          if(displayControls){
            dropdownGenre.silentOff();
            btnAuthorCloud.silentOff();
            btnPopularity.silentOff();
          }
          mode = "date";
          sortByDate();
          // TODO: PREVENT DRAGGING/SLIDING
        } else if(command[2].equals("genre")){
          if(displayControls){
            btnDate.silentOff();
            btnAuthorCloud.silentOff();
            btnPopularity.silentOff();
          }
          mode = "genre";
          //println("Displaying authors with "+ genreList_english.get(Integer.parseInt(command[3])) +" genre");
          sortByGenre((String)genreList_english.get(Integer.parseInt(command[3])));
          // TODO: PREVENT DRAGGING/SLIDING?
        } else if(command[2].equals("popularity")){
          if(displayControls){
            btnDate.silentOff();
            btnAuthorCloud.silentOff();
            dropdownGenre.silentOff();
          }
          mode = "popularity";
          sortByPopularity();
          // TODO: PREVENT DRAGGING/SLIDING
        } 
      }
    } else if(command[0].equals("quote")){    // affect the quote in some way
      if(command[1].equals("fadein")){
        // fade out any bio belonging to this author that exists first
        Iterator iter = bioObjects.values().iterator();
        while(iter.hasNext()){
          QuoteLine bioLine = (QuoteLine)iter.next();
          if(bioLine.getQuoteID() == Integer.parseInt(command[2])){
            bioLine.fadeOutAndRemove();
          }
        }
        Author author = (Author)textBlocks.get(Integer.parseInt(command[2]));
        if(author.hasQuote(userLanguage)){
          author.addControls(widgetManager, balloon);
          author.retrigger();
          createQuote(author);
        }
      }
    } else if(command[0].equals("bio")){      // affect the bio in some way
      if(command[1].equals("fadein")){
        // fade out any quote belonging to this author that exists first
        Iterator iter = quoteObjects.values().iterator();
        while(iter.hasNext()){
          QuoteLine quoteLine = (QuoteLine)iter.next();
          if(quoteLine.getQuoteID() == Integer.parseInt(command[2])){
            quoteLine.fadeOutAndRemove();
          }
        }
        Author author = (Author)textBlocks.get(Integer.parseInt(command[2]));
        if(author.hasBio(userLanguage)){
          author.addControls(widgetManager, balloon);
          author.retrigger();
          createBio(author);
        }
      }
    }
  }
  render(c);
}











// WIDGET BASED CONTROL FUNCTIONS

void redValue(float r){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setRed(r);
  }
}
void greenValue(float g){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setGreen(g);
  }
}
void blueValue(float b){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setBlue(b);
  }
}
void alphaMax(float a){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setAlphaMax(a);
  }
}
void alphaFallOff(float a){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setAlphaFallOff(a);
  }
}
void dragDamping(float d){
  dragDamp = d;
}





// USER INTERACTION FUNCTIONS

void mousePressed(){
  if(displayControls){
    widgetManager.pressed(mouseX, mouseY);
    //client.broadcast("press," + (mouseX+(client.getXoffset()*2)) + "," + mouseY);
    //println("sending " + int(mouseX * 1.6) +","+ int(mouseY * 1.6 - 320));
    if(!widgetManager.isOverAWidget(mouseX, mouseY)){
      //client.broadcast("press,"+ int(mouseX * interfaceScale) +","+ int((mouseY * interfaceScale) - verticalOffset));
      client.broadcast("press,"+ int((mouseX  + horizontalMouseOffset) * interfaceScale)+","+int((mouseY + verticalMouseOffset) * interfaceScale));
    }
  }
}

void mousePressedEvent(int xpos, int ypos){
  //println("mouse pressed: "+xpos +" "+ypos);
  boolean clickedBackground = true;
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    if(textBlock.isOver(xpos, ypos)){
      textBlock.press();
      clickedBackground = false;
    }
  }
  
  if(clickedBackground && displayControls){
    widgetManager.removeItem(balloon);
  }
  
  inactivityCounter = 0;  // keep track of last user action and count frames
}

void mouseReleased(){
  if(displayControls){
    widgetManager.released(mouseX, mouseY);
    if(!widgetManager.isOverAWidget(mouseX, mouseY)){
      //client.broadcast("release,"+ int(mouseX * interfaceScale) +","+ int((mouseY * interfaceScale) - verticalOffset));
       client.broadcast("release,"+ int((mouseX  + horizontalMouseOffset) * interfaceScale)+","+int((mouseY + verticalMouseOffset) * interfaceScale));
    }
  }
}

void mouseReleasedEvent(int xpos, int ypos){
  //println("mouse released: "+xpos +" "+ypos);
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    if(textBlock.isOver(xpos, ypos)){
      if(textBlock instanceof Author){
        //if(System.currentTimeMillis() - lastDragged > postDragDelay){
          if(((Author)textBlock).hasQuote(userLanguage)){
            if(!((Author)textBlock).triggered && textBlock.pressed){
              createQuote((Author)textBlock);  // create quote here based on author properties
              if(displayControls){
                widgetManager.removeItem(balloon);
                balloon = new Balloon("Balloon", (Author)textBlock, 0, balloonImage, balloonDown, buttonQuoteImage, buttonQuoteDown, buttonQuoteEspImage, buttonQuoteEspDown, buttonQuoteGrey, buttonBiographyImage, buttonBiographyDown, buttonBiographyEspImage, buttonBiographyEspDown, buttonBiographyGrey, interfaceScale, horizontalMouseOffset, verticalMouseOffset, userLanguage);
                widgetManager.addItem(balloon);
                ((Author)textBlock).addControls(widgetManager, balloon);
              }
              textBlock.release();
            } else {
              textBlock.releasedOutside();
            }
          }
          //textBlock.release();
        //} else {
        //  textBlock.releasedOutside();
        //}
      }
    } else {
      textBlock.releasedOutside();
    }
  }
}

void mouseDragged(){
  if(displayControls){
    widgetManager.dragged(mouseX, mouseY);
    if(!widgetManager.isOverAWidget(mouseX, mouseY)){
      int force = int(mouseX*interfaceScale) - int(pmouseX*interfaceScale);
      //client.broadcast("drag,"+ force +","+ int(mouseX*interfaceScale) +","+ ((int)(mouseY*interfaceScale) - verticalOffset));
       client.broadcast("drag,"+ force +","+ int((mouseX  + horizontalMouseOffset) * interfaceScale)+","+int((mouseY + verticalMouseOffset) * interfaceScale));
    }
  }
}

void keyPressed(){
  if(key == 'c' || key == 'C'){         // toggle gravity along the x axis
    if(displayCursor){
      noCursor();
    } else {
      cursor(ARROW);
    }
    displayCursor = !displayCursor;
  } else if(key == 'b' || key == 'B'){  // toggle bounding box display
    displayBoundingBoxes = !displayBoundingBoxes;
  } else if(key == 's' || key == 'S'){  // toggle statistics display
    displayStats = !displayStats;
  } else if(key == 'v' || key == 'V'){  // toggle camera feed
    enableCamera = !enableCamera;
  } else if(key == 'f' || key == 'F'){
    displayFrames = !displayFrames;
  } else if(key == 'y' || key == 'Y'){
    yflip = !yflip;
  }
}
