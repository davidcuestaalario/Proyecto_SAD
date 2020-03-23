package parte1;

public class PreProcesado 
{

	public static void main(String[] args) 
	{
		
		System.out.println("--------------------");
		System.out.println("INICIO PREPROCESADO");
		System.out.println("-------------------- \n");
		
		if( args.length == 0 ) 
		{			
			// RUTA
			GetRaw rawARFF = new GetRaw( "PRUEBAS" );
			TransformRaw wordVector = new TransformRaw( "PRUEBAS" );
			MakeCompatible remove = new MakeCompatible( "PRUEBAS" );
			
			// FICHERO
			rawARFF.getRawARFF( "train" );
			rawARFF.getRawARFF( "test_unk" );
			
			// FICHERO , BoW-TFIDF , Sparse-NonSparse
			wordVector.getWordVector( "train" , "BoW"   , "Sparse" );
			wordVector.getWordVector( "train" , "TFIDF" , "Sparse" );
			wordVector.getWordVector( "train" , "BoW"   , "NonSparse" );
			wordVector.getWordVector( "train" , "TFIDF" , "NonSparse" );
			
			// FICHERO , BoW-TFIDF , Sparse-NonSparse
			wordVector.getWordVector( "test_unk" , "BoW"   , "Sparse" );
			wordVector.getWordVector( "test_unk" , "TFIDF" , "Sparse" );
			wordVector.getWordVector( "test_unk" , "BoW"   , "NonSparse" );
			wordVector.getWordVector( "test_unk" , "TFIDF" , "NonSparse" );
			
			// BoW-TFIDF , Sparse-NonSparse
			wordVector.filtradoAtributos("BoW"   , "Sparse");
			wordVector.filtradoAtributos("TFIDF" , "Sparse");
			wordVector.filtradoAtributos("BoW"   , "NonSparse");
			wordVector.filtradoAtributos("TFIDF" , "NonSparse");
						
			// BoW-TFIDF , Sparse-NonSparse
			remove.RemoveTest("BoW"   , "Sparse");
			remove.RemoveTest("TFIDF" , "Sparse");
			remove.RemoveTest("BoW"   , "NonSparse");
			remove.RemoveTest("TFIDF" , "NonSparse");
			
			remove.compatibleTest("BoW"   , "Sparse");
			remove.compatibleTest("TFIDF" , "Sparse");
			remove.compatibleTest("BoW"   , "NonSparse");
			remove.compatibleTest("TFIDF" , "NonSparse");
		}
		else if( args.length == 3 ) 
		{
			
			String ruta = args[0];
			String bow = args[1];
			String sparse = args[2];
			
			// RUTA
			GetRaw rawARFF = new GetRaw( ruta );
			TransformRaw wordVector = new TransformRaw( ruta );
			MakeCompatible remove = new MakeCompatible( ruta );
			
			// FICHERO
			rawARFF.getRawARFF( "train" );
			rawARFF.getRawARFF( "test_unk" );
			
			// FICHERO , BoW-TFIDF , Sparse-NonSparse
			wordVector.getWordVector( "train"    , bow , sparse );
			wordVector.getWordVector( "test_unk" , bow , sparse );
			
			// BoW-TFIDF , Sparse-NonSparse
			wordVector.filtradoAtributos( bow , sparse );
					
			// BoW-TFIDF , Sparse-NonSparse
			remove.RemoveTest( bow , sparse );
			remove.compatibleTest( bow , sparse );
			
		}
		else 
		{
			System.out.println(" A continuacion se realizara el PreProcesado de los ficheros train y test_unk  ");
			System.out.println(" localizados en la ubicaion determinada por el primer parametro   ");
			System.out.println(" para determinar la configuracion se utilizaran el segundo y tercer parametro   ");
			System.out.println(" el segundo parametro tomara el valor de BoW o TFIDF para seleccionar el tipo de vectorizacion  ");
			System.out.println(" el tercer parametro tomara el valor de Sparse o NonSparse para seleccionar la representacion del vector   ");
			System.out.println("   ");
			System.out.println(" si no se utiliza ningun parametro se ejecutaran todas las convinaciones posibles en la carpeta DATA ");
		}
		
		System.out.println("\n--------------------");
		System.out.println("FIN PREPROCESADO");
		System.out.println("-------------------- \n");
	}

}