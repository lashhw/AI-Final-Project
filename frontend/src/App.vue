<script>
import AreaChart from './components/AreaChart.vue'
import info_json from './assets/info.json'

export default {
  data() {
    return {
      info: info_json,
      selection: {
        area: "",
        lot: ""
      },
      history: {},
      prediction: {},
      chart_series: []
    }
  },
  computed: {
    area_list() {
      return [...new Set(this.info.map(item => item.AREA))]
    },
    chart_options() {
      return {
        title : {
          text: `Parking Vacancy of ${this.selection.lot.name}`
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
  methods: {
    async update_series() {
      const res_history = await fetch(`https://ntpcparking.azurewebsites.net/api/gethistory?id=${this.selection.lot.id}&max_days=1`)
      this.history = await res_history.json()

      const res_prediction = await fetch(`https://ntpcparking.azurewebsites.net/api/predictfuture?id=${this.selection.lot.id}`)
      this.prediction = await res_prediction.json()

      var history_pairs = Object.entries(this.history)
      history_pairs.forEach(pair => pair[0] = Date.parse(`${pair[0]} GMT`))
      var midnight = new Date().setHours(0, 0, 0, 0)
      history_pairs = history_pairs.filter(pair => pair[0] >= midnight)

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
    }
  },
  components: { AreaChart }
}
</script>

<template>
  <select v-model="selection.area">
    <option value="" disabled>請選擇地區</option>
    <option v-for="area in area_list">
      {{ area }}
    </option>
  </select>
  <select v-model="selection.lot" @change="update_series">
    <option value="" disabled>請選擇停車場名稱</option>
    <template v-for="x in info">
      <option v-if="x.AREA === selection.area" :value="{ id: x.ID, name: x.NAME }">
        {{ x.NAME }}
      </option>
    </template>
  </select>
  <AreaChart :series="chart_series" :options="chart_options" />
</template>

<style>

</style>
