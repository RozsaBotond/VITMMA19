# Bull-flag detector

## 1. A projekt célja
A projekt célja egy olyan mélytanuló rendszer létrehozása, amely képes felismerni a pénzügyi idősorokban gyakran előforduló "bull flag" és "bear flag" mintázatokat. Ezek a minták egy hirtelen, markáns árfolyammozgásból (a "zászló rúdja") és egy azt követő, szűkülő tartományú konszolidációs időszakból (a "zászló") állnak. A modell feladata, hogy megkülönböztesse a különböző flag típusokat, és elkülönítse azokat a nem jellegzetes ármozgásoktól.

## 2. Mintázatok leírása
A feladat során két fő mintázat-irányt és azon belül három altípust kell megkülönböztetni.

● **Bull Flag (Bika Zászló):** Egy erős emelkedő trend ("rúd") után egy enyhén lefelé vagy oldalazva mozgó, szűkülő konszolidációs csatorna ("zászló") alakul ki. Ez jellemzően a trend folytatódását jelzi előre.

● **Bear Flag (Medve Zászló):** Egy erős csökkenő trend ("rúd") után egy enyhén felfelé vagy oldalazva mozgó, szűkülő konszolidációs csatorna ("zászló") alakul ki. Ez jellemzően a csökkenő trend folytatódását vetíti előre.

### Zászló altípusok
A konszolidációs szakasz (a "zászló") formája alapján három típust különböztetünk meg:

● **Normal:** A konszolidáció egy párhuzamos csatornában történik.
● **Wedge (Ék):** A konszolidációs sáv egy ék alakban szűkül.
● **Pennant / Straight (Árbócszalag):** A konszolidáció egy szimmetrikus, háromszög alakú formációban zajlik.

## 3. Adatgyűjtés

### Adatforrás
● **Eszközök:** EURUSD, XAUUSD, US30, US100, US500, GER40
● **Időfelbontások:** 1 perc, 5 perc, 15 perc, 30 perc, 1 óra
● **Lehetséges forrás:** A historikus adatok letöltéséhez például a polygon.io, kaggle.com, alphavantage.co, finnhub.io szolgáltatók használhatóak.

### Adatok előkészítése
1. Válassz ki egy instrumentumot és egy időfelbontást.
2. Keress az idősoron a 2. pontban leírtaknak megfelelő "flag" mintázatokat.
3. Mentsd ki a mintázatot tartalmazó idősor-szegmenst egy különálló .csv fájlba. A CSV fájlnak tartalmaznia kell az időbélyeget (Unix timestamp formátumban, UTC időzóna) és az adott időegységre vonatkozó Open, High, Low, Close (OHLC) értékeket. Példa CSV formátum:

```csv
timestamp,open,high,low,close
1672531200000,1.0704,1.0705,1.0703,1.0704
1672531260000,1.0704,1.0706,1.0704,1.0705
...
```

### Követelmények
● **Adatmennyiség:** Minimum 50 különböző flag mintázatot kell összegyűjteni és megcímkézni.

## 4. Fájlnevezési konvenció
Az összegyűjtött idősor-szegmenseket tartalmazó CSV fájlokat az alábbi séma szerint nevezd el:

`instrumentum_idofelbontas_sorszam.csv`

● **instrumentum:** Pl. EURUSD, US100.
● **idofelbontas:** Pl. 5min, 1H.
● **sorszam:** Három számjegyű sorszám (pl. 001, 002).

**Példák:**
● `EURUSD_15min_001.csv`
● `US500_1H_023.csv`

## 5. Adatfeltöltés és felhasználási jogok
Az összegyűjtött .csv fájlokat és az elkészült címkéket (.json) az alábbi helyre kell feltölteni:

**URL:** `https://bit.ly/bme-dl-pw-2025`

### Feltöltés menete:
1. Nyisd meg a fenti linket.
2. Keresd meg a `bullflagdetector` nevű könyvtárat.
3. A `bullflagdetector` könyvtáron belül hozz létre egy új alkönyvtárat a saját Neptun kódoddal.
4. Ebbe a `<NeptunKód>` nevű könyvtárba töltsd fel az összes forrás .csv fájlt és a Label Studio által generált címke fájlt.

**Fontos:** A feltöltéssel lemondasz az adatokkal kapcsolatos minden jogodról, és hozzájárulsz, hogy az általad gyűjtött és címkézett adatokat bárki szabadon felhasználhassa.

## 6. Címkézési útmutató (Label Studio)

### 6.1. Label Studio futtatása Dockerrel
Futtasd a Label Studio-t a már ismert módon, a .csv fájlokat tartalmazó mappádat csatolva:

```bash
docker run -it -p 8080:8080 -v <abszolút_útvonal_az_adatokhoz>:/label-studio/data heartexlabs/label-studio:latest
```

### 6.2. Projekt létrehozása és adatimport
1. Hozz létre egy új projektet (pl. `BullFlag_Detector_Cimkezes`).
2. Az **Data Import** résznél töltsd fel az előkészített .csv fájljaidat “Time Series or Whole Text File”-ként.

### 6.3. Címkézési felület beállítása
1. Menj a projekt **Settings > Labeling Interface** menüpontjára.
2. A kódszerkesztőbe másold be az alábbi XML kódot. Ez létrehozza a szükséges címkézési felületet az idősor-régió kijelöléséhez és a kategóriák megadásához.

```xml
<View>
  <Header>Válasszon egy címkét, majd jelölje ki a 'zászló' szakaszt az idősoron!</Header>
  <TimeSeriesLabels name="label" toName="ts">
    <Label value="Bullish Normal" background="#228B22"/>
    <Label value="Bullish Wedge" background="#006400"/>
    <Label value="Bullish Pennant" background="#32CD32"/>
    <Label value="Bearish Normal" background="#DC143C"/>
    <Label value="Bearish Wedge" background="#8B0000"/>
    <Label value="Bearish Pennant" background="#FF4500"/>
  </TimeSeriesLabels>
  <TimeSeries name="ts" valueType="url" value="$csv" sep="," timeColumn="timestamp" timeFormat="%Y-%m-%d %H:%M" timeDisplayFormat="%Y-%m-%d" overviewChannels="close">
    <Channel column="close" displayFormat=",.1f" strokeColor="#00FF00" legend="Close Price"/>
  </TimeSeries>
</View>
```
3. Kattints a **Save** gombra.

**TIPP:** ha elkészíted, hogy a címkézés során OHLC gyertyák jelenjenek meg (pl. TW LightWeight Charts integrációval), akkor az már önmagában +1 jegyet jelenthet.

### 6.4. A címkézés folyamata
1. A felületen megjelenik az idősor grafikonja.
2. Az egérrel húzva jelöld ki azt a régiót, amely a "zászló" konszolidációs szakaszt tartalmazza. A "zászló rúdja" ne legyen a kijelölés része.
3. A kijelölés után a jobb oldali panelen válaszd ki a mintázat irányát (Direction) és típusát (Type).
4. Kattints a **Submit** gombra.

### 6.5. Adatok exportálása
A címkézés végeztével a projekt oldalon kattints az **Export** gombra, és válaszd a **JSON** formátumot.

## 7. Az exportált annotációs fájl (JSON) formátuma
Az exportált fájl minden eleme egy-egy .csv fájlhoz tartozó annotációt tartalmaz. A feldolgozáshoz a `value.start` és `value.end` időbélyegek, valamint a `value.choices` tömbben lévő címkék a legfontosabbak.

Példa a JSON fájl szerkezetére:
```json
[
  {
    "file_upload": "EURUSD_15min_001.csv",
    "annotations": [
      {
        "result": [
          {
            "value": {
              "start": "2023-01-10T10:15:00.000Z",
              "end": "2023-01-10T11:45:00.000Z",
              "instant": false,
              "timeserieslabels": [
                "0_bullish",
                "Wedge"
              ]
            },
            "from_name": "direction",
            "to_name": "ts",
            "type": "timeserieslabels"
          }
        ]
      }
    ]
  }
]
```
