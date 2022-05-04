<template>
  <q-page padding>
    <div class="row flex-center q-col-gutter-sm">
      <div class="col-5">
        <q-select v-model="selection.area" :options="area_list" label="行政區" />
      </div>
      <div class="col-5">
        <q-select v-model="selection.lot" :options="p_lot_list" label="停車場名稱" />
      </div>
    </div>
    <AreaChart class="q-pa-md" :series="chart_series" height="50%" :options="chart_options" />
    <div class="row flex-center">
      <q-spinner-dots v-show="loading" color="primary" size="2em" />
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
        lot: {
          label: "",
          value: -1
        }
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
                      .map(x => ({ label: x.NAME, value: { id: x.ID, name: x.NAME } }))
    },
    chart_options() {
      return {
        title : {
          text: `Parking Vacancy of ${this.selection.lot.value.name}`
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
      this.update_series()
    }
  },
  methods: {
    async update_series() {
      this.loading = true;
      try {
        const res_history = await fetch(`https://ntpcparking.azurewebsites.net/api/gethistory?id=${this.selection.lot.value.id}&max_days=1`)
        this.history = await res_history.json()

        const res_prediction = await fetch(`https://ntpcparking.azurewebsites.net/api/predictfuture?id=${this.selection.lot.value.id}`)
        this.prediction = await res_prediction.json()

        var history_pairs = Object.entries(this.history)
        history_pairs.forEach(pair => pair[0] = Date.parse(`${pair[0]} GMT`))
        var now = Date.now()
        history_pairs = history_pairs.filter(pair => pair[0] >= now - 43200000)

        var last_history_time = history_pairs[history_pairs.length-1][0]
        var prediction_pairs = Object.entries(this.prediction)
        prediction_pairs.forEach(pair => pair[0] = Date.parse(`${pair[0]} GMT`))
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