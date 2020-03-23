package Proyecto_Preguntas;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Attribute;
import weka.core.DictionaryBuilder;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.NonSparseToSparse;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class PreProcesado 
{
	
	private String ruta;
	
	public PreProcesado( String pRuta ) { this.ruta = pRuta; }
	
	// CONVIERTE A FORMATO ARFF EL FICHERO DE REGUNTAS EN FORMATO CSV.
	public void getRawARFF( String pNombre ) 
	{
		String datoscsv = this.ruta + "/" + pNombre + ".csv"; 
		String rawarff = this.ruta + "/" + pNombre + "_RaW.arff";
		String tmparf = this.ruta + "/" + pNombre + "_RaW_SinComillas.arff";
		String tmp = ruta + "/ztmp.csv";
    	
    	try
        {
        	System.out.println("Iniciando conversion del fichero " + pNombre + "...");
        	
        	BufferedReader br = new BufferedReader(new FileReader( datoscsv ));
        	BufferedWriter bw = new BufferedWriter(new FileWriter( tmp ));
			
        	String linea;
        	while((linea = br.readLine()) != null)
        	{
        		// ELIMINAR CUALQUIER CARACTER NO ASCII DE LAS INSTANCIAS
        		linea = linea.replaceAll("[^\\p{ASCII}]", "");
        		linea = linea.replaceAll("\\.", "");
        		linea = linea.replaceAll("\\?", "");
        		linea = linea.replaceAll("\\:", "");
        		linea = linea.replaceAll("\\-", "");
        		linea = linea.replaceAll("\\&", "");
        		linea = linea.replaceAll("'", "");
        		linea = linea.replaceAll("`", "");
        			
        		// DIVIDIR EN TRES PARTES PARA SEPARAR LOS ATRIBUTOS
        		String[] partes = linea.split(",", 3);
        		
        		// ESCRIBIR EL RESULTADO
        		bw.write(partes[2]);
        		bw.write(" , ");
        		bw.write(partes[1]);
        		
                bw.newLine();		
        	}
        	br.close();
        	bw.close();
        	System.out.println("   Fichero temporal limpio...");
        	
        	// CARGAR EL CSV LIMPIO Y MARCAR COMO STRING LA PREGUNTA
           	CSVLoader loader = new CSVLoader();
            loader.setSource( new File( tmp ) );
            loader.setStringAttributes("last");
        	Instances data = loader.getDataSet();    	
        	System.out.println("   Fichero temporal cargado...");
        	
        	// SETTEAR NOMBRE DE LA RELACION
        	data.setRelationName("preguntas_" + pNombre);
        	System.out.println("   Nombre de la relacion establecida...");
        	
        	// GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
        	BufferedWriter writer = new BufferedWriter(new FileWriter( rawarff ));
        	writer.write(data.toString());
        	writer.flush();
        	writer.close();
        	System.out.println("Conversion del fichero " + pNombre + " finalizada... \n");
        	 
        	// QUITAR COMILLAS
        	BufferedReader read = new BufferedReader(new FileReader( rawarff ));
        	BufferedWriter write = new BufferedWriter(new FileWriter( tmparf ));
			
        	String line;
        	int encabezado = 0;
        	while( (line = read.readLine()) != null )
        	{
        		encabezado++;
        		if(encabezado < 5) 
        		{
        			line = line.replaceAll("'", "");
            		line = line.replaceAll("`", "");
        		}
        		write.write(line);
        		write.newLine();
        	}
        	write.close();
        }
        catch(Exception e) { e.printStackTrace(); }
	}
	
	// APLICA LA VECTORIZACION BoW o TF-IDF COMPRIMIDO COMO Sparse o NonSparse
	public void getWordVector( String pNombre , String pBow , String pSparse ) 
	{
		String name = pNombre;
	
		boolean sparse = true;
		if( pSparse == "NonSparse" ) { sparse = false; }
    	boolean bow = true;
    	if( pBow == "TFIDF" ) { bow = false; }
    	
		String config = "";
		
		if ( bow ) { config += "_BoW_"; }
    	else { config += "_TFIDF_"; }
    	
    	if ( sparse ) { config += "Sparse"; }
    	else { config += "NonSparse"; }
    		    	
    	String rawarff = this.ruta + "/" + name + "_RaW.arff";
    	String bowarff = this.ruta + "/" + name + config + ".arff";   	
    	String diccionario = this.ruta + "/" + name + config + "_Diccionario.txt";
    	
		try
	    {
			System.out.println("Iniciando la conversion " + config + "...");
			
			// CARGAMOS EL CONJUNTO DE DATOS DEL RAW
			Instances data = new DataSource( rawarff ).getDataSet();
			
			// TOKENICER
			NGramTokenizer tokenizer = new NGramTokenizer();
			tokenizer.setNGramMaxSize(1);
			tokenizer.setNGramMinSize(1);
			tokenizer.setDelimiters( " \n 	.,;'\"()?!-/<>‘’“”…«»•&{[|`^]}$*%" );
		    
			// APLICAMOS EL FILTRO StringToWordVector
	    	StringToWordVector filtroBow = new StringToWordVector();
	    	filtroBow.setInputFormat( data );
	    	filtroBow.setOutputWordCounts( !bow );  
	    	filtroBow.setIDFTransform( !bow );
	    	filtroBow.setTFTransform( !bow );
	    	filtroBow.setLowerCaseTokens(true);
	    	filtroBow.setTokenizer( tokenizer );
	    	filtroBow.setDictionaryFileToSaveTo(new File( diccionario ));
	    	
	    	data = Filter.useFilter(data, filtroBow);
	    	System.out.println("   Filtro StringToWordVector aplicado...");
	    	
	    	// APLICAMOS EL FILTRO NonSparseToSparse
	    	if ( !sparse ) 
	    	{
	    		NonSparseToSparse filtroSparse = new NonSparseToSparse();
           	 	filtroSparse.setInputFormat(data);
           	 	data = Filter.useFilter(data, filtroSparse);
           	 	System.out.println("   Filtro SparseToSparsede aplicado...");
	    	}
	    	else
	    	{
	    		SparseToNonSparse filtroSparse = new SparseToNonSparse();
           	 	filtroSparse.setInputFormat(data);
           	 	data = Filter.useFilter(data, filtroSparse);
           	 	System.out.println("   Filtro NonSparseToSparsede aplicado...");
	    	}
	    	
	    	// MOVEMOS LA CLASE A PREDECIR AL FINAL
			Reorder filtroR = new Reorder();
			filtroR.setAttributeIndices("2-last,1");
			filtroR.setInputFormat(data);
			System.out.println("   Clase a predecir movida al ultimo atributo...");
							           	 	
			// SETTEAR NOMBRE DE LA RELACION
        	data.setRelationName("preguntas_" + name + "_" + config);
        	System.out.println("   Nombre de la relacion establecida...");
        	           	 
	        // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
       	 	BufferedWriter writer = new BufferedWriter(new FileWriter( bowarff ));
       	 	writer.write(data.toString());
       	 	writer.flush();
       	 	writer.close();
       	 	System.out.println("Conversion " + config + " finalizada... \n");
	    } 
		catch (Exception e) { e.printStackTrace(); }
	}
	
	// FILTRA LOS ATRIBUTOS QUE NO CORRELAN CON LA CLASE Y SI CORRELAN ENTRE SI
	public void filtradoAtributos( String pBow , String pSparse ) 
	{		
		boolean sparse = true;
		if( pSparse == "NonSparse" ) { sparse = false; }
    	boolean bow = true;
    	if( pBow == "TFIDF" ) { bow = false; }
    	
		String train_original = this.ruta + "/train_" + pBow + "_" + pSparse + ".arff" ;
		String train_resultado = this.ruta + "/train_" + pBow + "_" + pSparse + "_Filtrados.arff";
		String diccionario = this.ruta + "/train_" + pBow + "_" + pSparse + "_Diccionario_Filtrado.txt";
		
		try
		{						
			// FILTRAMOS LOS DATOS DE ENTRENAMIENTO
			System.out.println("Iniciando el filtrado de los datos de entrenamiento " + pBow + "_" + pSparse + "...");
			Instances dataTrain = new DataSource( train_original ).getDataSet();
			dataTrain.setClassIndex(0);
						
			// APLICAMOS LA SELECCION DE ATRIBUTOS AL TRAIN EN FUNCION DE LA CORRELACION	        
	        AttributeSelection filtro = new AttributeSelection();
			CfsSubsetEval cfs = new CfsSubsetEval();
			BestFirst search = new BestFirst();
				
			filtro.setSearch(search);
			filtro.setEvaluator(cfs);
			filtro.setInputFormat( dataTrain );
			
			dataTrain = filtro.useFilter( dataTrain , filtro);
			System.out.println("   Datos de entrenamiento filtrados...");
			
			//DictionaryBuilder dictionaryBuilder = new DictionaryBuilder();
			//dictionaryBuilder.setup( dataTrain );
			//dictionaryBuilder.setMinTermFreq(1);
			//dictionaryBuilder.finalizeDictionary();
			//dictionaryBuilder.saveDictionary( new File( diccionario ) , false );
			
			// SETTEAR NOMBRE DE LA RELACION
			dataTrain.setRelationName("preguntas_" + pBow + "_" + pSparse + "_Filtrados");
        	System.out.println("   Nombre de la relacion establecida...");
        	           	 
	        // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
       	 	BufferedWriter trainWriter = new BufferedWriter( new FileWriter( train_resultado ) );
       	 	trainWriter.write( dataTrain.toString() );
       	 	trainWriter.flush();
       	 	trainWriter.close();
       	 	System.out.println("   Filtrado finalizado... \n");
		}
		catch(Exception e) { e.printStackTrace(); }
	}
	
	// CONVIERTE EL FICHERO DE TEST AL MISMO FORMATO QUE EL DE TRAIN UTILIZANDO REMOVE
	public void RemoveTest( String pBow , String pSparse ) 
	{
		String train_filtrado = this.ruta + "/train_" + pBow + "_" + pSparse + "_Filtrados.arff";
		String test_original = this.ruta + "/test_unk_" + pBow + "_" + pSparse + ".arff";
		String test_resultado = this.ruta + "/test_unk_" + pBow + "_" + pSparse + "_Filtrados.arff";
		
		try 
		{
			System.out.println("Iniciando la compatibilizacion de los datos de test " + pBow + "_" + pSparse + "...");
			
			BufferedReader encabezado = new BufferedReader(new FileReader( train_filtrado ));
	    	BufferedReader cuerpo = new BufferedReader(new FileReader( test_original ));
	    	BufferedWriter write = new BufferedWriter(new FileWriter( test_resultado ));
				
	    	DataSource trainSource = new DataSource( train_filtrado );
	    	DataSource testSource = new DataSource( test_original );
	    	Instances train = trainSource.getDataSet();
	    	Instances test = testSource.getDataSet();
		   	if (train.classIndex() == -1) { train.setClassIndex(train.numAttributes() - 1); }
		   	if (test.classIndex() == -1) { test.setClassIndex(test.numAttributes() - 1); }
		    
		    int[] indicesANoEliminar = new int[train.numAttributes()-1];
			
			ArrayList<Attribute> atributosPorDefecto = new ArrayList<Attribute>();
			Enumeration<Attribute> at = train.enumerateAttributes();
			while(at.hasMoreElements()) 
			{
				atributosPorDefecto.add(at.nextElement());
			}
				
			int i=0;
			Enumeration<Attribute> at2 = train.enumerateAttributes();
			while(at2.hasMoreElements()) 
			{
				Attribute atributo = at2.nextElement();
				if(atributosPorDefecto.contains(atributo)) 
				{
					indicesANoEliminar[i] = atributosPorDefecto.indexOf(atributo);
					i++;
				}				
			}
			
			Remove remove = new Remove();
			remove.setAttributeIndicesArray(indicesANoEliminar);
			remove.setInvertSelection(true);
			remove.setInputFormat(train);
			test = Filter.useFilter(test, remove);
			    
		    // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
		    write.write( test.toString() );
		    write.flush();
		    write.close();
		    System.out.println("   Filtrado finalizado... \n");
		    
			} 
			catch(Exception e) { e.printStackTrace(); }			 
		}
		
	// FILTRA LOS ATRIBUTOS QUE NO CORRELAN CON LA CLASE Y SI CORRELAN ENTRE SI
	public void filtradoBarridoAtributos( String pBow , String pSparse ) 
	{		
		boolean sparse = true;
		if( pSparse == "NonSparse" ) { sparse = false; }
    	boolean bow = true;
    	if( pBow == "TFIDF" ) { bow = false; }
    	
		String train_original = this.ruta + "/train_" + pBow + "_" + pSparse + ".arff" ;
		String train_resultado = this.ruta + "/train_" + pBow + "_" + pSparse + "_Filtrados.arff";
		String diccionario = this.ruta + "/train_" + pBow + "_" + pSparse + "_Diccionario_Filtrado.txt";
		
		try
		{						
			// FILTRAMOS LOS DATOS DE ENTRENAMIENTO
			System.out.println("Iniciando el filtrado de los datos de entrenamiento " + pBow + "_" + pSparse + "...");
			Instances dataTrain = new DataSource( train_original ).getDataSet();
			dataTrain.setClassIndex(0);
			
			Discretize discretice = new Discretize();
			discretice.setInputFormat( dataTrain );
			dataTrain = discretice.useFilter( dataTrain , discretice );
						
			// APLICAMOS LA SELECCION DE ATRIBUTOS AL TRAIN	        
			Remove filtro = new Remove();
			filtro.setInvertSelection( true );
			boolean parar = false;
			double precisionActual = 0;
			double precisionAnterior = 0;
			int numeroAtributos = 0;
			
			// AUMENTAMOS EL LIMITE DE ATRIBUTOS DE 100 EN 100
			for(int i = 100; i < dataTrain.numAttributes() - 1 && !parar; i += 100)
			{				
				filtro.setAttributeIndices("first-" + i + ", last");
				filtro.setInputFormat( dataTrain );

		        Instances dataReducida = Filter.useFilter( dataTrain , filtro );
		        dataReducida.setClassIndex( dataReducida.numAttributes() - 1 );
		        
		        // ENTRENAMOS UN CLASIFICADOR
		        NaiveBayesMultinomial clasificador = new NaiveBayesMultinomial(); 
				Evaluation evaluador = new Evaluation( dataReducida );
				evaluador.crossValidateModel(clasificador , dataReducida , 10 , new Random(1));
				precisionActual = evaluador.pctCorrect();
				
				System.out.println("     - Numero de atributos: " + dataReducida.numAttributes() );
				System.out.println("     - Accuracy: " + precisionAnterior );
				
				// ACTUALIZAMOS HASTA QUE LA PRECISION EMPIECE A REDUCIRSE EN CULLO CASO PARAMOS 
				if( precisionActual > precisionAnterior )
				{
					precisionAnterior = precisionActual;
					numeroAtributos = i;
				}
				else { parar = true; }
			}
				
			filtro.setAttributeIndices("first-" + numeroAtributos + ", last");
			filtro.setInputFormat( dataTrain );
			dataTrain = Filter.useFilter( dataTrain , filtro );	        
			System.out.println("   Numero de atributos finales: " + dataTrain.numAttributes());
			
			// SETTEAR NOMBRE DE LA RELACION
			dataTrain.setRelationName("preguntas_" + pBow + "_" + pSparse + "_Filtrados");
        	System.out.println("   Nombre de la relacion establecida...");
        	           	 
	        // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
       	 	BufferedWriter trainWriter = new BufferedWriter( new FileWriter( train_resultado ) );
       	 	trainWriter.write( dataTrain.toString() );
       	 	trainWriter.flush();
       	 	trainWriter.close();
       	 	System.out.println("   Filtrado finalizado... \n");
		}
		catch(Exception e) { e.printStackTrace(); }
	}
	
	// CONVIERTE EL FICHERO DE TEST AL MISMO FORMATO QUE EL DE TRAIN UTILIZANDO FixedDictionaryStringToWordVector
	public void FixedTest( String pBow , String pSparse ) 
	{
		boolean sparse = true;
		if( pSparse == "NonSparse" ) { sparse = false; }
    	boolean bow = true;
    	if( pBow == "TFIDF" ) { bow = false; }
		
		String test_original = this.ruta + "/test_unk_RaW.arff";
		String test_resultado = this.ruta + "/test_unk_" + pBow + "_" + pSparse + "_Filtrados.arff";
		String diccionario = this.ruta + "/test_unk_" + pBow + "_" + pSparse + "_Filtrados_Diccionario.txt";
		
		try 
		{
			// FILTRAMOS LOS DATOS DE ENTRENAMIENTO
	   		System.out.println("Iniciando la compatibilizacion de los datos del test " + pBow + "_" + pSparse + "...");
	   		Instances dataTest = new DataSource( test_original ).getDataSet();
	   		
	   	    FixedDictionaryStringToWordVector fixer = new FixedDictionaryStringToWordVector();      	 	
	   	    fixer.setOutputWordCounts( !bow );  
	   	    fixer.setIDFTransform( !bow );
	   	    fixer.setTFTransform( !bow );
	   	    fixer.setLowerCaseTokens(true);
	   	    fixer.setDictionaryFile(new File( diccionario ));
	   	    fixer.setInputFormat( dataTest );
	   	    dataTest = Filter.useFilter( dataTest , fixer);
	   	    System.out.println("   Filtro StringToWordVector aplicado...");
	   	    
	   	    if ( !sparse ) 
	    	{
	    		NonSparseToSparse filtroSparse = new NonSparseToSparse();
	    	 	filtroSparse.setInputFormat(dataTest);
	    	 	dataTest = Filter.useFilter(dataTest, filtroSparse);
	    	 	System.out.println("   Filtro NonSparseToSparsede aplicado...");
	    	}
	    	else
	    	{
	    		SparseToNonSparse filtroSparse = new SparseToNonSparse();
	    	 	filtroSparse.setInputFormat(dataTest);
	    	 	dataTest = Filter.useFilter(dataTest, filtroSparse);
	    	 	System.out.println("   Filtro SparseToNonSparse aplicado...");
	    	}
	   	    
	   	    // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
	   	 	BufferedWriter testWriter = new BufferedWriter( new FileWriter( test_resultado ) );
	   	 	testWriter.write( dataTest.toString() );
	   	 	testWriter.flush();
	   	 	testWriter.close();
	   	 	System.out.println("   Filtrado finalizado... \n");
		}
		catch(Exception e) { e.printStackTrace(); }   	 
	}

	// CONVIERTE EL FICHERO DE TEST AL MISMO FORMATO QUE EL DE TRAIN CON FUERZA BRUTA 
	public void compatibleTest( String pBow , String pSparse ) 
	{
		boolean sparse = true;
		if( pSparse == "NonSparse" ) { sparse = false; }
    	boolean bow = true;
    	if( pBow == "TFIDF" ) { bow = false; }

		String test_original = this.ruta + "/test_unk_" + pBow + "_" + pSparse + ".arff";
		String test_resultado = this.ruta + "/test_unk_" + pBow + "_" + pSparse + "_Filtrados.arff";
		String train_original = this.ruta + "/train_" + pBow + "_" + pSparse + "_Filtrados.arff";
		
		try 
		{
			System.out.println("Inicindo la conversion de los datos del test para el formato " + pBow + "_" + pSparse + "...");
			
	    	BufferedReader encabezado = new BufferedReader(new FileReader( train_original ));
	    	BufferedReader cuerpo = new BufferedReader(new FileReader( test_original ));
	    	BufferedWriter write = new BufferedWriter(new FileWriter( test_resultado ));
			
	    	String word = "@data";
	    	
	    	// COPIAMOS EL ENCABEZADO DEL TRAIN
	    	String lineCabeza;
	    	boolean fin = false;
	    	while( (lineCabeza = encabezado.readLine()) != null )
	    	{
	    		if ( !fin ) 
	    		{
	    			//System.out.println( lineCabeza.contains( word ) );
		    		//System.out.println(lineCabeza);
	    			write.write( lineCabeza );
		    		write.newLine();
	    		}
	    		if( lineCabeza.contains( word ) ) { fin = true; }
	    	}
	    	
	    	// COPIAMOS LOS DATOS DEL TEST
	    	String lineCuerpo;
	    	while( (lineCuerpo = cuerpo.readLine()) != null )
	    	{
	    		if ( !fin ) 
	    		{
	    			//System.out.println( lineCabeza.contains( word ) );
		    		//System.out.println(lineCabeza);
	    			write.write( lineCuerpo );
		    		write.newLine();
	    		}
	    		if( lineCuerpo.contains( word ) ) { fin = false; }
	    	}
	    	write.close();
		}
		catch(Exception e) { e.printStackTrace(); }
		
	}
}
