<script>
import AreaChart from './components/AreaChart.vue'

export default {
  data() {
    return {
      id: null,
      history: {},
      prediction: {},
      chart_options: {
        xaxis: {
          type: 'datetime'
        },
        yaxis: {
          decimalsInFloat: 0
        },
        dataLabels: {
          enabled: false
        },
        tooltip: {
          x: {
            format: 'dd MMM hh:mm TT'
          }
        }
      }
    }
  },
  computed: {
    chart_series() {
      return [
        {
          name: "history",
          data: Object.entries(this.history)
        },
        {
          name: "prediction",
          data: Object.entries(this.prediction)
        }
      ]
    }
  },
  methods: {
    async get_data() {
      const res_history = await fetch(`https://ntpcparking.azurewebsites.net/api/gethistory?id=${this.id}&max_days=0`)
      this.history = await res_history.json()

      const res_prediction = await fetch(`https://ntpcparking.azurewebsites.net/api/predictfuture?id=${this.id}`)
      this.prediction = await res_prediction.json()
    }
  },
  components: { AreaChart }
}
</script>

<template>
  <input v-model="id" placeholder="停車場 ID" />
  <button @click="get_data"> 送出 </button>
  <AreaChart :series="chart_series" :options="chart_options" />
</template>

<style>

</style>
