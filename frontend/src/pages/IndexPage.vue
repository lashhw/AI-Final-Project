<template>
  <q-page padding>
    <div class="q-gutter-md">
      <div class="row flex-center q-gutter-sm">
        <div class="col-5">
          <q-select v-model="selection.area" :options="area_list" label="行政區" />
        </div>
        <div class="col-5">
          <q-select v-model="selection.lot" :options="p_lot_list" label="停車場名稱" />
        </div>
      </div>
      <div class="row flex-center">
        <div class="col-12 col-xs-10 col-sm-8 col-md-6 col-lg-4">
          <AreaChart :series="chart_series" :options="chart_options" />
        </div>
      </div>
      <div class="row flex-center">
        <div class="col-auto">
          <q-spinner-dots v-show="loading" color="primary" size="2em" />
        </div>
      </div>
    </div>
  </q-page>
</template>

<script>
import AreaChart from '../components/AreaChart.vue'
import info_json from '../assets/info.json'

export default {
  name: 'IndexPage',
  data() {
    return {
      info: info_json,
      selection: {
        area: "",
        lot: ""
      },
      history: {},
      prediction: {},
      chart_series: [],
      loading: false
    }
  },
  computed: {
    area_list() {
      return [...new Set(this.info.map(item => item.AREA))]
    },
    p_lot_list() {
      return this.info.filter(x => x.AREA === this.selection.area)
                      .map(x => x.NAME)
    },
    chart_options() {
      return {
        title : {
          text: `Parking Vacancy of ${this.selection.lot}`
        },
        chart: {
          toolbar: {
            offsetY: 20
          }
        },
        xaxis: {
          type: "datetime",
          labels: {
            datetimeUTC: false
          }
        },
        yaxis: {
          decimalsInFloat: 0
        },
        stroke: {
          dashArray: [0, 4]
        },
        dataLabels: {
          enabled: false
        },
        tooltip: {
          x: {
            format: "dd MMM hh:mm TT"
          }
        },
        annotations: {
          xaxis: [
            {
              x: Date.now(),
              label: {
                text: "now"
              }
            }
          ]
        }
      }
    }
  },
  watch: {
    'selection.lot'() {
      for (var i = 0; i < this.info.length; i++) {
        if (this.info[i].NAME === this.selection.lot) {
          this.update_series(this.info[i].ID)
          return
        }
      }
    }
  },
  methods: {
    async update_series(id) {
      this.loading = true;
      try {
        const res_history = await fetch(`https://ntpcparking.azurewebsites.net/api/gethistory?id=${id}&max_days=1`)
        this.history = await res_history.json()

        const res_prediction = await fetch(`https://ntpcparking.azurewebsites.net/api/predictfuture?id=${id}`)
        this.prediction = await res_prediction.json()

        var history_pairs = Object.entries(this.history)
        for (var i = 0; i < history_pairs.length; i++) {
          var splitted = history_pairs[i][0].split(/[- :]/)
          history_pairs[i][0] = Date.UTC(splitted[0], splitted[1], splitted[2], splitted[3], splitted[4])
        }
        var now = Date.now()
        history_pairs = history_pairs.filter(pair => pair[0] >= now - 43200000)

        var prediction_pairs = Object.entries(this.prediction)
        for (var i = 0; i < prediction_pairs.length; i++) {
          var splitted = prediction_pairs[i][0].split(/[- :]/)
          prediction_pairs[i][0] = Date.UTC(splitted[0], splitted[1], splitted[2], splitted[3], splitted[4])
        }
        var last_history_time = history_pairs[history_pairs.length-1][0]
        prediction_pairs = prediction_pairs.filter(pair => pair[0] > last_history_time)
      
        this.chart_series = [
          {
            name: 'History',
            data: history_pairs
          },
          {
            name: 'LSTM prediction',
            data: prediction_pairs
          }
        ]
      } catch (error) {
        // error handling
      }
      this.loading = false;
    }
  },
  components: { AreaChart }
}
</script>